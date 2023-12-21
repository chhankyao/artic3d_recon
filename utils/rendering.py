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


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.structures import Meshes
from pytorch3d.ops import interpolate_face_attributes
from pytorch3d.renderer import (
    look_at_rotation,
    look_at_view_transform,
    PerspectiveCameras,
    OrthographicCameras,
    AmbientLights,
    PointLights,
    BlendParams,
    RasterizationSettings,
    MeshRenderer,
    MeshRendererWithFragments,
    MeshRasterizer,
    SoftSilhouetteShader,
    SoftPhongShader,
    HardPhongShader,
    Textures,
    TexturesVertex,
    Materials,
)
from pytorch3d.renderer.mesh.shading import phong_shading, flat_shading, gouraud_shading
from config import cfg
from data_utils import *


def layered_rgb_blend(colors, fragments, blend_params, clip_inside=True, debug=False):
    N, H, W, K = fragments.pix_to_face.shape
    device = fragments.pix_to_face.device
    pixel_colors = torch.ones((N, H, W, 4), dtype=colors.dtype, device=colors.device)
    background_ = blend_params.background_color
    if not isinstance(background_, torch.Tensor):
        background = torch.tensor(background_, dtype=torch.float32, device=device)
    else:
        background = background_.to(device)
    mask = fragments.pix_to_face >= 0
    if blend_params.sigma == 0:
        alpha = (fragments.dists <= 0).float() * mask
    elif clip_inside:
        alpha = torch.exp(-fragments.dists.clamp(0) / blend_params.sigma) * mask
    else:
        alpha = torch.sigmoid(-fragments.dists / blend_params.sigma) * mask
    occ_alpha = torch.cumprod(1.0 - alpha, dim=-1)
    occ_alpha = torch.cat([torch.ones(N, H, W, 1, device=device), occ_alpha], dim=-1)
    colors = torch.cat([colors, background[None, None, None, None].expand(N, H, W, 1, -1)], dim=-2)
    alpha = torch.cat([alpha, torch.ones(N, H, W, 1, device=device)], dim=-1)
    pixel_colors[..., :3] = (occ_alpha[..., None] * alpha[..., None] * colors).sum(-2)
    pixel_colors[..., 3] = 1 - occ_alpha[:, :, :, -1]
    return pixel_colors


class LayeredShader(nn.Module):
    def __init__(self, device='cpu', cameras=None, lights=None, materials=None, blend_params=None, clip_inside=True,
                 shading_type='phong', debug=False):
        super().__init__()
        self.lights = lights if lights is not None else DirectionalLights(device=device)
        self.materials = (materials if materials is not None else Materials(device=device))
        self.cameras = cameras
        self.blend_params = blend_params if blend_params is not None else BlendParams()
        self.clip_inside = clip_inside
        if shading_type == 'phong':
            shading_fn = phong_shading
        elif shading_type == 'flat':
            shading_fn = flat_shading
        elif shading_type == 'gouraud':
            shading_fn = gouraud_shading
        elif shading_type == 'raw':
            shading_fn = lambda x: x
        else:
            raise NotImplementedError
        self.shading_fn = shading_fn
        self.shading_type = shading_type
        self.debug = debug

    def to(self, device):
        cameras = self.cameras
        if cameras is not None:
            self.cameras = cameras.to(device)
        self.materials = self.materials.to(device)
        self.lights = self.lights.to(device)
        return self

    def forward(self, fragments, meshes, **kwargs):
        blend_params = kwargs.get("blend_params", self.blend_params)
        if self.shading_type == 'raw':
            colors = meshes.sample_textures(fragments)
            if not torch.all(self.lights.ambient_color == 1):
                colors *= self.lights.ambient_color
        else:
            sh_kwargs = {'meshes': meshes, 'fragments': fragments, 'cameras': kwargs.get("cameras", self.cameras),
                         'lights': kwargs.get("lights", self.lights),
                         'materials': kwargs.get("materials", self.materials)}
            if self.shading_type != 'gouraud':
                sh_kwargs['texels'] = meshes.sample_textures(fragments)
            colors = self.shading_fn(**sh_kwargs)
        return layered_rgb_blend(colors, fragments, blend_params, clip_inside=self.clip_inside, debug=self.debug)
    
    
class Renderer():
    def __init__(self, device, shader, light):
        super().__init__()
        self.device = device
        self.shader = shader
        self.elevs = torch.tensor([0, 0, -30, 30]).float().to(device)
        self.azims = torch.tensor([0, 90]).float().to(device)
        R, T = look_at_view_transform(dist=5, elev=0, azim=0, device=device)
        self.cams = OrthographicCameras(device=device, focal_length=1, R=R, T=T)
        self.part_color = torch.zeros(cfg.nb,3).float().to(device)
        for i in range(cfg.nb):
            self.part_color[i,:] = torch.tensor(part_colors[i+1][:3]).to(device)

        # Light settings
        if light == 'ambient':
            self.lights = AmbientLights(device=device)
            
        elif light == 'point':
            self.lights = PointLights(
                device=device, location=[[0.0, 1.0, 2.0]], ambient_color=((0.5,0.5,0.5),),
                diffuse_color=((0.3,0.3,0.3),), specular_color=((0.2,0.2,0.2),)
            )
        
        # Shader settings
        if shader == 'soft_sil':
            blend_params = BlendParams(sigma=1e-4, gamma=1e-4, background_color=(0.0,0.0,0.0)) 
            raster_settings = RasterizationSettings(
                image_size=cfg.input_size,
                blur_radius=np.log(1./1e-4-1.)*blend_params.sigma,
                faces_per_pixel=50,
                bin_size=0,
            )
            shader = SoftSilhouetteShader(blend_params=blend_params)
            
        elif shader == 'soft_phong':
            blend_params = BlendParams(sigma=1e-4, gamma=1e-4, background_color=(0.0,0.0,0.0)) 
            raster_settings = RasterizationSettings(
                image_size=cfg.input_size,
                blur_radius=np.log(1./1e-4-1.)*blend_params.sigma*0.5,
                faces_per_pixel=50,
                bin_size=0,
            )
            shader = LayeredShader(device=device, cameras=self.cams[0], lights=self.lights, blend_params=blend_params)
            
        else:
            raster_settings = RasterizationSettings(
                image_size=cfg.input_size,
                blur_radius=0.0,
                faces_per_pixel=1,
                bin_size=0,
                max_faces_per_bin=100,
            )
            shader = HardPhongShader(device=device, cameras=self.cams[0], lights=self.lights)
            
        self.renderer = MeshRendererWithFragments(
            rasterizer=MeshRasterizer(cameras=self.cams[0], raster_settings=raster_settings),
            shader=shader
        ).to(device)
    
    def render(self, verts, faces, verts_rgb=None, part_idx=-1, mv=False, rand=False, fragment=False):
        bs = verts.shape[0]
        if len(verts.shape) == 3:
            nv = verts.shape[1]
            verts_combined = verts
            faces_combined = faces[None,:,:].repeat(bs,1,1)
        else:
            nb, nv = verts.shape[1], verts.shape[2]
            verts_combined = verts.permute(0,3,1,2).reshape(bs,3,-1).permute(0,2,1)
            faces_combined = torch.cat([faces + i*nv for i in range(nb)], 0)[None,:,:].repeat(bs,1,1)

        if self.shader == 'soft_sil':
            mesh = Meshes(verts=verts_combined, faces=faces_combined)
        else:
            if verts_rgb is None:
                if part_idx == -1:
                    verts_rgb = self.part_color[None,:,None,:].repeat(bs,1,nv,1)
                else:
                    verts_rgb = self.part_color[None,part_idx,None,:].repeat(bs,1,nv,1)
            verts_rgb = verts_rgb.permute(0,3,1,2).reshape(bs,3,-1).permute(0,2,1)
            mesh = Meshes(verts=verts_combined, faces=faces_combined, textures=TexturesVertex(verts_rgb))
            
        if mv:
            if rand:
                elev = torch.tensor([-15 + np.random.rand()*45]).float().to(self.device)
                azim = np.random.rand()*360
                azim = torch.tensor([azim, azim+90]).float().to(self.device)
                focal = torch.tensor([1.0 + np.random.rand()*0.2]).float().to(self.device)
                R, T = look_at_view_transform(dist=5, elev=elev, azim=azim, device=self.device)
                cams = OrthographicCameras(device=self.device, focal_length=focal, R=R, T=T)
                sigma = self.renderer.shader.blend_params.sigma
                gamma = self.renderer.shader.blend_params.gamma
                blend_params = BlendParams(sigma=sigma, gamma=gamma, background_color=torch.rand(3).float().to(self.device)) 
                self.renderer.shader.blend_params = blend_params
                imgs, fragments = self.renderer(mesh.extend(2), cameras=cams)
            else:
                R, T = look_at_view_transform(dist=5, elev=0, azim=self.azims, device=self.device)
                cams = OrthographicCameras(device=self.device, focal_length=1, R=R, T=T)
                sigma = self.renderer.shader.blend_params.sigma
                gamma = self.renderer.shader.blend_params.gamma
                blend_params = BlendParams(sigma=sigma, gamma=gamma, background_color=(1.0,1.0,1.0)) 
                self.renderer.shader.blend_params = blend_params
                imgs, fragments = self.renderer(mesh.extend(R.shape[0]), cameras=cams)
        else:
            imgs, fragments = self.renderer(mesh, cameras=self.cams[0])
        
        if fragment:
            return imgs, fragments, mesh
        else:
            return imgs

    def project(self, x):
        return self.cams[0].transform_points_screen(x, image_size=cfg.input_size)
    
    def set_sigma_gamma(self, sigma, gamma):
        blend_params = BlendParams(sigma=sigma, gamma=gamma, background_color=(0.0,0.0,0.0)) 
        self.renderer.shader.blend_params = blend_params
        self.renderer.rasterizer.raster_settings.blur_radius = np.log(1./1e-4-1.)*sigma*0.5
    
    def get_part_masks(self, verts, faces):
        bs, nb, nv = verts.shape[:3]
        masks = []
        for i in range(bs):
            verts_combined = verts[i:i+1].permute(0,3,1,2).reshape(1,3,-1).permute(0,2,1)
            faces_combined = torch.cat([faces + i*nv for i in range(nb)], 0)[None]
            meshes = Meshes(verts=verts_combined, faces=faces_combined)
            fragments = self.renderer.rasterizer(meshes, cameras=self.cams[0])
            mask = torch.div(fragments.pix_to_face, faces.shape[0], rounding_mode='floor')+1  # (1, H, W, 1)
            mask[fragments.pix_to_face == -1] = 0
            masks.append(mask)
        return torch.cat(masks, 0).permute(0,3,1,2)
    
    def get_verts_vis(self, verts, faces):
        bs, nb, nv = verts.shape[:3]
        verts_vis = []
        for i in range(bs):
            verts_combined = verts[i:i+1].permute(0,3,1,2).reshape(1,3,-1).permute(0,2,1)
            faces_combined = torch.cat([faces + i*nv for i in range(nb)], 0)[None]
            meshes = Meshes(verts=verts_combined, faces=faces_combined)
            packed_faces = meshes.faces_packed() 
            pix_to_face = self.renderer.rasterizer(meshes, cameras=self.cams[0]).pix_to_face # (1, H, W, 1)
            visible_faces = pix_to_face.unique()
            visible_verts = torch.unique(packed_faces[visible_faces])
            visibility_map = torch.zeros_like(verts_combined[0,:,0])
            visibility_map[visible_verts] = 1
            verts_vis.append(visibility_map.view(nb, nv))
        return torch.stack(verts_vis, 0)
