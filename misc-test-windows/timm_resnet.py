'''
Author: wangwei83 wangwei83@cuit.edu.cn
Date: 2024-06-02 22:06:32
LastEditors: wangwei83 wangwei83@cuit.edu.cn
LastEditTime: 2024-06-05 22:38:23
FilePath: /wangwei/X-23d-Y-ai-Z-detection/misc-test-windows/timm_resnet.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''

# https://huggingface.co/timm/resnetv2_152x2_bit.goog_teacher_in21k_ft_in1k_384
# https://github.com/Alibaba-MIIL/ImageNet21K 
# ImageNet-1K 数据集是深度学习模型进行计算机视觉任务预训练的主要数据集。
# ImageNet-21K 数据集包含更多的图片和类别，由于其复杂性而导致使用较少。


import timm
from timm.models.efficientnet import _cfg
import torch

config = _cfg(url='', file='/data/wangwei/resnetv2_152x2_bit.bin') #file为本地文件路径

model = timm.create_model(
                                "resnet34",
                                pretrained=True,
                                features_only=True,
                                pretrained_cfg=config
                            )

x = torch.randn(1,3,224,224)
output = model(x)
# output.shape
print(output)
# 测试