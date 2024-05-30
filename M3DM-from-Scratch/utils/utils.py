'''
Author: wangwei83 wangwei83@cuit.edu.cn
Date: 2024-05-31 00:18:23
LastEditors: wangwei83 wangwei83@cuit.edu.cn
LastEditTime: 2024-05-31 00:19:17
FilePath: /wangwei/X-23d-Y-ai-Z-detection/M3DM-from-Scratch/utils/utils.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''


import torch

class KNNGaussianBlur(torch.nn.Module):
    def __init__(self,radius:int =4):
        super().__init__()
    