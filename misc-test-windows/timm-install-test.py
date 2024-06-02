'''
Author: wangwei83 wangwei83@cuit.edu.cn
Date: 2024-05-31 23:45:25
LastEditors: wangwei83 wangwei83@cuit.edu.cn
LastEditTime: 2024-06-02 21:33:26
FilePath: /wangwei/X-23d-Y-ai-Z-detection/misc-test-windows/timm-install-test.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''

# 参考代码：https://blog.csdn.net/weixin_48846514/article/details/138920374
import timm

model= timm.create_model('resnet18', pretrained=True, 
            pretrained_cfg_overlay=dict(file='/data/wangwei/X-23d-Y-ai-Z-detection/misc-test-windows/pytorch_model.bin'))
print(model)

