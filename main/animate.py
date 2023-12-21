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


import os.path as osp
import json
from argparse import ArgumentParser
from config import cfg
from dataloader import *
from model import *
from data_utils import *


def optimize_animation():        
    print("========== Loading data of %s... ========== " % cfg.animal_class)
    num_imgs, inputs = load_data()

    print("========== Preparing model... ========== ")
    skeleton_path = osp.join(cfg.model_dir, 'skeleton_%s.json' % cfg.animal_class)
    model = Model(cfg.device, num_imgs, skeleton_path)
    model.load_model(osp.join(cfg.model_dir, '%s.pth'%cfg.animal_class))
    model.load_parts(osp.join(cfg.model_dir, '%s_part_%d.pth' % (cfg.animal_class, cfg.instance_idx)))
    model.load_texture(osp.join(cfg.model_dir, '%s_text_%d.pth' % (cfg.animal_class, cfg.instance_idx)))
    model.eval()
    
    print("========== Fine-tuning animation... ========== ")
    uvs, faces = model.meshes[5].get_uvs_and_faces()
    inputs['uvs'], inputs['faces'] = uvs, faces
    outputs = model.forward(inputs, stop_at=10, text='uv')        
    verts_color = outputs['verts_color']
    global_scale, global_trans, global_rot, bone_rot = model.get_instance_params(idx=0)
    
    # get source animation frames
    imgs = []
    verts_2d_all = []
    pix_to_face_all = []
    for v in range(16):
        w = v/7.5
        bone_rot2 = bone_rot * (1-w)
        rot = torch.cat([model.rot_id[None,None,:], bone_rot2 + model.bone_rot_rest[None,:,:]], 1)
        joints_can, joints_rot = model.skeleton.transform_joints(rot, scale=model.bone_scale)
        verts_can = model.transform_verts(uvs, joints_can, joints_rot, True, 10)
        verts = model.global_transform(verts_can.reshape(1,-1,3), global_rot, global_trans, global_scale)
        verts_2d = model.hard_renderer.project(verts)/cfg.input_size[0]
        img, fragments, mesh = model.hard_renderer.render(verts.reshape(1,cfg.nb,-1,3), faces, verts_color, fragment=True)
        verts_2d_all.append(verts_2d)
        pix_to_face_all.append(fragments.pix_to_face.flatten())
        imgs.append(img[...,:3].permute(0,3,1,2))
    imgs_rev = imgs[1:-1].copy()
    imgs_rev.reverse()
    save_img('animation_source_%d.gif'%(cfg.instance_idx), [img2np(img.permute(0,2,3,1)) for img in imgs + imgs_rev])
    
    # calculate forward flow
    flow_forward = []
    for v in range(1,16):
        pix_to_verts = mesh.faces_packed()[pix_to_face_all[v]] # N x 3            
        flow = verts_2d_all[v-1][0,pix_to_verts.flatten()].view(-1,3,3).mean(1) # N x 3
        flow[...,2] = 1
        flow[pix_to_face_all[v]==-1,:2] = 0
        flow[pix_to_face_all[v]==-1,2] = 0
        flow = flow.permute(1,0).view(1,3,cfg.input_size[1],cfg.input_size[0])
        flow = F.interpolate(flow, (64,64), mode='bilinear', align_corners=False)
        flow_forward.append(flow)
    flow_forward = torch.cat(flow_forward, 0)
    flow_forward[:,:2] = flow_forward[:,:2]*2-1
    flow_forward = flow_forward.permute(0,2,3,1).detach() # B-1 x 64 x 64 x 3
        
    # calculate backward flow
    flow_backward = []
    for v in range(15):
        pix_to_verts = mesh.faces_packed()[pix_to_face_all[v]] # N x 3            
        flow = verts_2d_all[v+1][0,pix_to_verts.flatten()].view(-1,3,3).mean(1) # N x 3
        flow[...,2] = 1
        flow[pix_to_face_all[v]==-1,:2] = 0
        flow[pix_to_face_all[v]==-1,2] = 0
        flow = flow.permute(1,0).view(1,3,cfg.input_size[1],cfg.input_size[0])
        flow = F.interpolate(flow, (64,64), mode='bilinear', align_corners=False)
        flow_backward.append(flow)
    flow_backward = torch.cat(flow_backward, 0)
    flow_backward[:,:2] = flow_backward[:,:2]*2-1
    flow_backward = flow_backward.permute(0,2,3,1).detach() # B-1 x 64 x 64 x 3
        
    # T-DASS optimization
    with torch.no_grad():
        text_prompt = 'A photo of %s in white background' % cfg.animal_class
        text_z = model.diffusion.get_text_embeds([text_prompt])
        latents = [model.diffusion.encode_imgs(img) for img in imgs]
        
    latents = nn.Parameter(torch.cat(latents, 0))
    optimizer = torch.optim.Adam([latents], lr=1e-2)
    losses = {'dass':[], 'temp':[]}
    for j in tqdm(range(50)):
        optimizer.zero_grad()
        with torch.no_grad():
            latents_target = [model.diffusion.update_latents(text_z, l[None].clone(), g=12, t=250, n=3) for l in latents]
            latents_target = torch.cat(latents_target, 0)
        loss_dass = ((latents - latents_target)**2).mean()
        loss_temp = 0
        if j <= 45:
            latents_forward = F.grid_sample(latents[:-1], flow_forward[...,:2], mode='bilinear', align_corners=False)
            latents_backward = F.grid_sample(latents[1:], flow_backward[...,:2], mode='bilinear', align_corners=False)
            loss_temp = (((latents[1:] - latents_forward)**2) * flow_forward[...,2:].permute(0,3,1,2)).mean()
            loss_temp += (((latents[:-1] - latents_backward)**2) * flow_backward[...,2:].permute(0,3,1,2)).mean()
        loss = loss_dass + loss_temp * 0.05
        loss.backward()
        optimizer.step()
        losses['dass'].append(loss_dass)
        losses['temp'].append(loss_temp)
        
    # decode final animation
    imgs = [model.diffusion.decode_latents(l[None]).clamp(0,1) for l in latents]
    imgs = [img2np(img.permute(0,2,3,1)) for img in imgs]
    imgs_rev = imgs[1:-1].copy()
    imgs_rev.reverse()
    save_img('animation_output_%d.gif'%(cfg.instance_idx), imgs + imgs_rev)
    

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--cls', type=str, default='zebra', dest='cls')
    parser.add_argument('--inst', type=bool, default=False, dest='opt_instance')
    parser.add_argument('--idx', type=int, default=0, dest='instance_idx')
    args = parser.parse_args()
    cfg.set_args(args)

    optimize_animation()
