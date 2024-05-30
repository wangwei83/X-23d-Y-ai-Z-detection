'''
Author: wangwei83 wangwei83@cuit.edu.cn
Date: 2024-05-31 00:18:23
LastEditors: wangwei83 wangwei83@cuit.edu.cn
LastEditTime: 2024-05-31 00:24:14
FilePath: /wangwei/X-23d-Y-ai-Z-detection/M3DM-from-Scratch/utils/utils.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''


from typing import Any
import torch
from torchvision import transforms
from PIL import ImageFilter

class KNNGaussianBlur(torch.nn.Module):
    def __init__(self,radius:int =4):
        super().__init__()
        self.radius = radius
        self.unload = transforms.ToPILImage()
        self.load = transforms.ToTensor()
        self.blur_kernel = ImageFilter.GaussianBlur(radius=4)
    
    def __call__(self, img):
        map_max = img.max()
        final_map = self.load(self.unload(img[0]/map_max).filter(self.blur_kernel))*map_max
        return final_map
    