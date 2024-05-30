'''
Author: wangwei83 wangwei83@cuit.edu.cn
Date: 2024-05-27 21:34:17
LastEditors: wangwei83 wangwei83@cuit.edu.cn
LastEditTime: 2024-05-30 23:48:22
FilePath: /wangwei/X-23d-Y-ai-Z-detection/M3DM-from-Scratch/feature_extractors/multiple_features.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import os
from feature_extractors.features import Features

import numpy as np
from utils.mvtec3d_util import *

class DoubleRGBPointFeatures(Features):
    def add_sample_to_mem_bank(self, sample,class_name=None):
        organized_pc = sample[1]
        organized_pc_np = organized_pc.squeeze().permute(1, 2, 0).numpy()
        unorganized_pc = organized_pc_to_unorganized_pc(organized_pc=organized_pc_np)
        nonzero_indices = np.nonzero(np.all(unorganized_pc != 0, axis=1))[0]
        
        unorganized_pc_no_zeros = torch.tensor(unorganized_pc[nonzero_indices, :]).unsqueeze(dim=0).permute(0, 2, 1)
        rgb_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx, interpolated_pc = self(sample[0],unorganized_pc_no_zeros.contiguous())

       
        # xyz_patch = torch.cat(xyz_feature_maps,1)
        # xyz_patch_full = torch.zeros((1,interpolated_pc.shape[1],self.image_size*self.image_size),dtype=xyz_patch.dtype)
        # xyz_patch_full[:,:,nonzero_indices] = interpolated_pc
        # xyz_patch_full_2d = xyz_patch_full.view(1,interpolated_pc.shape[1],self.image_size,self.image_size)
        # xyz_patch_full_resized = self.resize(self.average(xyz_patch_full_2d))
        # xyz_patch=xyz_patch_full_resized.reshape(xyz_patch_full_resized.shape[1],-1).T
        
        # rgb_patch = torch.cat(rgb_feature_maps,1)
        # rgb_patch = rgb_patch.reshape(rgb_patch.shape[1],-1).T
        
        # rgb_patch_resize = rgb_patch.repeat(4,1).reshape(784,4,-1).permute(1,0,2).reshape(784*4,-1)
        
        # patch = torch.cat([xyz_patch,rgb_patch_resize],dim=1)
        
        # if class_name is not None:
        #     torch.save(patch,os.path.join(self.args.save_feature_path, class_name+str(self.ins_id) + '.pt'))
        #     self.ins_id += 1
        
        # self.patch_xyz_lib.append(xyz_patch)
        # self.patch_rgb_lib.append(rgb_patch)
        
    def predict(self, sample,mask,label):
        pass
    
    def __call__(self,sample0,unorganized_pc_no_zeros): 
        # 在这里实现你的逻辑
        rgb_feature_maps = ...
        xyz_feature_maps = ...
        center = ...
        neighbor_idx = ...
        center_idx = ...
        interpolated_pc = ...
        return rgb_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx, interpolated_pc