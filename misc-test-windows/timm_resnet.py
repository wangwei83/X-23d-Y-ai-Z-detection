'''
Author: wangwei83 wangwei83@cuit.edu.cn
Date: 2024-06-02 22:06:32
LastEditors: wangwei83 wangwei83@cuit.edu.cn
LastEditTime: 2024-06-04 21:53:09
FilePath: /wangwei/X-23d-Y-ai-Z-detection/misc-test-windows/timm_resnet.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import timm
from timm.models.efficientnet import _cfg
import torch


# avail_pretrained_models = timm.list_models(pretrained=True)
# len(avail_pretrained_models)
# print(len(avail_pretrained_models))

# all_densnet_models = timm.list_models("*densenet*")
# print(all_densnet_models)

config = _cfg(url='', file='/data/wangwei/resnetv2_152x2_bit.bin') #file为本地文件路径

model = timm.create_model(
                                "resnet34",
                                pretrained=True,
                                features_only=True,
                                pretrained_cfg=config
                            )

x = torch.randn(1,3,224,224)
output = model(x)
output.shape
# 测试