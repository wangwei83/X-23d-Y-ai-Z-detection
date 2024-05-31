'''
Author: wangwei83 wangwei83@cuit.edu.cn
Date: 2024-05-31 00:01:07
LastEditors: wangwei83 wangwei83@cuit.edu.cn
LastEditTime: 2024-05-31 11:32:50
FilePath: /wangwei/X-23d-Y-ai-Z-detection/M3DM-from-Scratch/models/models.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''

import torch
import timm


class Model(torch.nn.Module):
    
    def __init__(self,device,rgb_backbone_name='vit_base_patch8_224_dino',out_indices=None,checkpoint_path='',
                 pool_last=False,xyz_backbone_name='Point_MAE',group_size=128,num_group=1024):
        super().__init__()
        self.device = device
        
        kwargs={'features_only':True if out_indices is None else False}
        if out_indices:
            kwargs.update({'out_indices':out_indices})
            
        self.rgb_backbone = timm.create_model(model_name=rgb_backbone_name,pretrained=True,checkpoint_path=checkpoint_path,**kwargs)
       
        