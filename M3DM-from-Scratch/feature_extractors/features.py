'''
Author: wangwei83 wangwei83@cuit.edu.cn
Date: 2024-05-27 21:34:34
LastEditors: wangwei83 wangwei83@cuit.edu.cn
LastEditTime: 2024-05-31 00:07:09
FilePath: /wangwei/X-23d-Y-ai-Z-detection/M3DM-from-Scratch/feature_extractors/features.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''

import torch
from models.models import Model


class Features(torch.nn.Module):
    def __init__(self,args,image_size=224,f_coreset=0.1,coreset_eps=0.9):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.deep_feature_extractor = Model(
                                        device=self.device,
                                        rgb_backbone_name=args.rgb_backbone_name,
                                        xyz_backbone_name=args.xyz_backbone_name,
                                        group_size=args.group_size,
                                        num_group=args.num_group
                                        )
