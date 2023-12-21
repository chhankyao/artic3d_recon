# Copyright 2023 Chun-Han Yao
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     https://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from huggingface_hub import hf_hub_download
from torchvision import transforms
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModel, CLIPProcessor, logging
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler, DDIMScheduler
# suppress partial model loading warning
logging.set_verbosity_error()

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
import time


class StableDiffusion(nn.Module):
    def __init__(self, device, model_name='stabilityai/stable-diffusion-2-1-base'):
        super().__init__()
        self.device = device
        logger.info(f'loading stable diffusion with {model_name}...')
        
        # Create model
        self.vae = AutoencoderKL.from_pretrained(model_name, subfolder="vae").to(self.device)
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(model_name, subfolder="text_encoder").to(self.device)
        self.unet = UNet2DConditionModel.from_pretrained(model_name, subfolder="unet").to(self.device)
        # self.scheduler = DDIMScheduler.from_pretrained(model_name, subfolder="scheduler")
        self.scheduler = PNDMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", 
                                       num_train_timesteps=1000)
        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.min_step = int(self.num_train_timesteps * 0.02)
        self.max_step = int(self.num_train_timesteps * 0.5)
        self.alphas = self.scheduler.alphas_cumprod.to(self.device)
        logger.info(f'\t successfully loaded stable diffusion!')

    def get_text_embeds(self, prompt):
        # Tokenize text and get embeddings
        text_input = self.tokenizer(prompt, padding='max_length', max_length=self.tokenizer.model_max_length, 
                                    truncation=True, return_tensors='pt')

        with torch.no_grad():
            text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]

        # Do the same for unconditional embeddings
        uncond_input = self.tokenizer(['']*len(prompt), padding='max_length', 
                                      max_length=self.tokenizer.model_max_length, return_tensors='pt')

        with torch.no_grad():
            uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]

        # Cat for final embeddings
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        return text_embeddings
    
    def train_step(self, text_embeddings, inputs, weight=1.0, g=15, t=None, n=1):
        # encode image into latents with vae, requires grad!
        pred_rgb = F.interpolate(inputs, (512,512), mode='bilinear', align_corners=False)
        latents = self.encode_imgs(pred_rgb)

        # timestep ~ U(0.02, 0.5) to avoid very high/low noise level
        if t is None:
            t = torch.randint(self.min_step, self.max_step+1, [1], dtype=torch.long, device=self.device)
        else:
            t = torch.tensor([t]).long().to(self.device) 

        grad_latent = 0
        for _ in range(n):
            # predict the noise residual with unet, NO grad!
            with torch.no_grad():
                # add noise
                noise = torch.randn_like(latents)
                latents_noisy = self.scheduler.add_noise(latents, noise, t)
                
                # pred noise
                latent_model_input = torch.cat([latents_noisy]*2)
                noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

            # perform guidance (high scale from paper!)
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + g * (noise_pred_text - noise_pred_uncond)

            # w(t), alpha_t * sigma_t^2
            w = (1 - self.alphas[t])
            grad = w * (noise_pred - noise)

            # clip grad for stable training?
            grad = grad.clamp(-1,1)
            grad = torch.nan_to_num(grad) * weight
            
            # accumulate latent gradients
            grad_latent = grad_latent + grad

        # manually backward, since we omitted an item in grad and cannot simply autodiff.
        latents.backward(gradient=grad_latent/n, retain_graph=True)
        return 0 # dummy loss value
    
    def get_img_grad(self, text_embeddings, inputs, weight=1.0, g=15, t=None, n=1):
        # encode image into latents with vae, requires grad!
        pred_rgb = F.interpolate(inputs, (512,512), mode='bilinear', align_corners=False)
        with torch.no_grad():
            latents = self.encode_imgs(pred_rgb)

        # timestep ~ U(0.02, 0.5) to avoid very high/low noise level
        if t is None:
            t = torch.randint(self.min_step, self.max_step+1, [1], dtype=torch.long, device=self.device)
        else:
            t = torch.tensor([t]).long().to(self.device) 

        # predict the noise residual with unet, NO grad!
        with torch.no_grad():
            for i in range(n):
                # add noise
                noise = torch.randn_like(latents)
                latents_noisy = self.scheduler.add_noise(latents, noise, t)
                
                # pred noise
                latent_model_input = torch.cat([latents_noisy]*2)
                noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
                
                # perform guidance (high scale from paper!)
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + g * (noise_pred_text - noise_pred_uncond)

                # w(t), alpha_t * sigma_t^2
                w = (1 - self.alphas[t])
                grad = w * (noise_pred - noise)

                # clip grad for stable training?
                grad = grad.clamp(-1,1)
                grad = torch.nan_to_num(grad)
                
                # update latents
                latents -= grad*0.5

            # decode updated latents and calculate pixel gradients
            dec = self.decode_latents(latents).clamp(0,1)
            grad_img = (pred_rgb - dec) * weight
            
        # manually backward, since we omitted an item in grad and cannot simply autodiff.
        pred_rgb.backward(gradient=grad_img, retain_graph=True)
        return 0 # dummy loss value
    
    def get_img_target(self, text_embeddings, inputs, weight=1.0, g=15, t=None, n=1):
        # encode image into latents with vae, requires grad!
        pred_rgb = F.interpolate(inputs, (512,512), mode='bilinear', align_corners=False)
        with torch.no_grad():
            latents = self.encode_imgs(pred_rgb)

        # timestep ~ U(0.02, 0.5) to avoid very high/low noise level
        if t is None:
            t = torch.randint(self.min_step, self.max_step+1, [1], dtype=torch.long, device=self.device)
        else:
            t = torch.tensor([t]).long().to(self.device) 

        # predict the noise residual with unet, NO grad!
        with torch.no_grad():
            for _ in range(n):
                # add noise
                noise = torch.randn_like(latents)
                latents_noisy = self.scheduler.add_noise(latents, noise, t)
                
                # pred noise
                latent_model_input = torch.cat([latents_noisy]*2)
                noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

                # perform guidance (high scale from paper!)
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + g * (noise_pred_text - noise_pred_uncond)

                # w(t), alpha_t * sigma_t^2
                w = (1 - self.alphas[t])
                grad = w * (noise_pred - noise)

                # clip grad for stable training?
                grad = grad.clamp(-1,1)
                grad = torch.nan_to_num(grad)
                
                # update latents
                latents -= grad*0.5

            # decode updated latents
            dec = self.decode_latents(latents).clamp(0,1)
            
        return dec
    
    def update_latents(self, text_embeddings, latents, weight=1.0, g=15, t=100, n=1):
        # timestep ~ U(0.02, 0.5) to avoid very high/low noise level
        t = torch.tensor([t]).long().to(self.device) 

        # predict the noise residual with unet, NO grad!
        with torch.no_grad():
            for _ in range(n):
                # add noise
                noise = torch.randn_like(latents) 
                latents_noisy = self.scheduler.add_noise(latents, noise, t)
                
                # pred noise
                latent_model_input = torch.cat([latents_noisy]*2)
                noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

                # perform guidance (high scale from paper!)
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + g * (noise_pred_text - noise_pred_uncond)

                # w(t), alpha_t * sigma_t^2
                w = (1 - self.alphas[t])
                grad = w * (noise_pred - noise)

                # clip grad for stable training?
                grad = grad.clamp(-1,1)
                grad = torch.nan_to_num(grad)
                
                # update latents
                latents -= grad*0.5

        return latents

    def decode_latents(self, latents):
        # latents = F.interpolate(latents, (64,64), mode='bilinear', align_corners=False)
        latents = 1 / 0.18215 * latents
        with torch.no_grad():
            imgs = self.vae.decode(latents).sample
        imgs = (imgs / 2 + 0.5)
        return imgs

    def encode_imgs(self, imgs):
        # imgs = F.interpolate(imgs, (512,512), mode='bilinear', align_corners=False)
        imgs = 2 * imgs - 1
        posterior = self.vae.encode(imgs).latent_dist
        latents = posterior.sample() * 0.18215
        return latents
    
    def enhance_img(self, text_embeddings, img):
        img = nn.Parameter(img.clone())
        optimizer = torch.optim.SGD([img], lr=1.0, momentum=0.0)
        optimizer.zero_grad()
        _ = self.get_img_grad(text_embeddings, img, weight=1.0, g=12, t=100, n=10)
        # _ = self.get_img_grad(text_embeddings, img, weight=1.0, g=15, t=250, n=30)
        optimizer.step()
        return img.detach()
    
    def produce_latents(self, text_embeddings, latents, num_inference_steps=50, g=7.5):
        self.scheduler.set_timesteps(num_inference_steps)
        # latents = torch.randn_like(latents)
        
        with torch.autocast('cuda'):
            for i, t in enumerate(self.scheduler.timesteps):
            # for i, t in enumerate(range(500, 0, -10)):
                # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
                latent_model_input = torch.cat([latents]*2)

                # predict the noise residual
                with torch.no_grad():
                    noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings)['sample']

                # perform guidance
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + g * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents)['prev_sample']
        
        return latents
