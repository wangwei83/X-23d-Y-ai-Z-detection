'''
Author: wangwei83 wangwei83@cuit.edu.cn
Date: 2024-06-02 22:06:32
LastEditors: wangwei83 wangwei83@cuit.edu.cn
LastEditTime: 2024-06-03 18:51:41
FilePath: /wangwei/X-23d-Y-ai-Z-detection/misc-test-windows/timm_resnet.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import timm
avail_pretrained_models = timm.list_models(pretrained=True)
len(avail_pretrained_models)
print(len(avail_pretrained_models))

all_densnet_models = timm.list_models("*densenet*")
print(all_densnet_models)

model = timm.create_model('resnet34',num_classes=10,pretrained=True)
print(model.default_cfg)