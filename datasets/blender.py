import torch
from torch.utils.data import Dataset
import json
import numpy as np
import os
from PIL import Image
from torchvision import transforms as T
from einops import rearrange

from .ray_utils import *


class BlenderDataset(Dataset):
    def __init__(self, root_dir, split='train', img_wh=(800, 800)):
        self.root_dir = root_dir
        self.split = split
        assert img_wh[0] == img_wh[1], 'image width must equal image height!'
        self.img_wh = img_wh
        self.define_transforms()
        self.white_back = False
        self.back_mask = False
        self.read_meta()

    def read_meta(self):
        with open(os.path.join(self.root_dir,
                               f"transforms_{self.split}.json"), 'r') as f:
            self.meta = json.load(f)

        w, h = self.img_wh
        self.focal = 0.5 * 800/np.tan(0.5*self.meta['camera_angle_x'])  # original focal length when W=800
        self.focal *= self.img_wh[0]/800  # modify focal length to match size self.img_wh

        # bounds, common for all scenes
        self.near = 2.0
        self.far = 7.0
        # self.near = 12.0 # Simulated scene
        # self.far = 36.0
        self.bounds = np.array([self.near, self.far])
        
        # ray directions for all pixels, same for all images (same H, W, focal)
        self.directions = get_ray_directions(h, w, self.focal)  # (h, w, 3)
            
        if self.split == 'train':  # create buffer of all rays and rgb data
            self.image_paths = []
            self.poses = []
            self.all_rays = []
            self.all_rgbs = []
            self.all_masks = []

            self.semantic_image_paths = []
            self.all_semantics = []

            for frame in self.meta['frames']:
                pose = np.array(frame['transform_matrix'])[:3, :4]
                self.poses += [pose]
                c2w = torch.FloatTensor(pose)

                image_path = os.path.join(self.root_dir, f"{frame['file_path']}.png")
                semantic_image_path = os.path.join(self.root_dir, f"{frame['file_path'].replace('train','train_edge')}.png")
                self.image_paths += [image_path]
                self.semantic_image_paths += [semantic_image_path]

                img = Image.open(image_path).convert("RGBA")
                img = img.resize(self.img_wh, Image.Resampling.LANCZOS)
                img = self.transform(img)
                mask = img[3, ...].view(-1, 1)  # background 0
                # mask = self.transform(np.zeros(self.img_wh)).view(-1, 1)
                if self.white_back:
                    img = img[:3, ...] * img[-1, ...] + (1. - img[-1, ...])
                else:
                    img = img[:3, ...] * img[-1, ...]
                img = rearrange(img, 'n1 n2 n3 -> (n2 n3) n1')

                semantic_img = Image.open(semantic_image_path).convert('L')
                semantic_img = semantic_img.resize(self.img_wh, Image.Resampling.LANCZOS)
                semantic_img = self.transform(semantic_img)  # (1, h, w)
                semantic_img = semantic_img.view(-1, 1)

                self.all_rgbs += [img]
                self.all_semantics += [semantic_img]
                self.all_masks += [mask]
                
                rays_o, rays_d = get_rays(self.directions, c2w)  # both (h*w, 3)

                self.all_rays += [torch.cat([rays_o, rays_d, 
                                             self.near*torch.ones_like(rays_o[:, :1]),
                                             self.far*torch.ones_like(rays_o[:, :1])],
                                             1)] # (h*w, 8)

            self.all_rays = torch.cat(self.all_rays, 0)     # (len(self.meta['frames])*h*w, 3)
            self.all_rgbs = torch.cat(self.all_rgbs, 0)     # (len(self.meta['frames])*h*w, 3)
            self.all_semantics = torch.cat(self.all_semantics, 0)  # (len(self.meta['frames])*h*w, 1)
            self.all_masks = torch.cat(self.all_masks, 0)

    def define_transforms(self):
        self.transform = T.ToTensor()

    def __len__(self):
        if self.split == 'train':
            return len(self.all_rays)
        if self.split == 'val':
            return 4
        return len(self.meta['frames'])

    def __getitem__(self, idx):
        if self.split == 'train': # use data in the buffers
            sample = {'rays': self.all_rays[idx],
                      'rgbs': self.all_rgbs[idx],
                      'edges': self.all_semantics[idx]}
                      # 'masks': self.all_masks[idx]}

        else: # create data for each image separately
            frame = self.meta['frames'][idx]
            c2w = torch.FloatTensor(frame['transform_matrix'])[:3, :4]

            img_path = os.path.join(self.root_dir, f"{frame['file_path']}.png")
            semantic_image_path = os.path.join(self.root_dir,
                                               f"{frame['file_path'].replace('val', 'val_edge')}.png")

            edge_img = Image.open(semantic_image_path).convert('L')
            edge_img = edge_img.resize(self.img_wh, Image.Resampling.LANCZOS)
            edge_img = self.transform(edge_img)
            edge_img = edge_img.view(-1, 1)

            img = Image.open(img_path).convert("RGBA")
            img = img.resize(self.img_wh, Image.Resampling.LANCZOS)
            img = self.transform(img)
            mask = img[3, ...].view(-1, 1)
            if self.white_back:
                img = img[:3, ...] * img[-1, ...] + (1. - img[-1, ...])
            else:
                img = img[:3, ...] * img[-1, ...]
            img = rearrange(img, 'n1 n2 n3 -> (n2 n3) n1')

            rays_o, rays_d = get_rays(self.directions, c2w)

            rays = torch.cat([rays_o, rays_d, 
                              self.near*torch.ones_like(rays_o[:, :1]),
                              self.far*torch.ones_like(rays_o[:, :1])],
                              1)  # (H*W, 8)

            sample = {'rays': rays, 'rgbs': img, 'c2w': c2w, 'edges': edge_img, 'mask': mask}
        return sample