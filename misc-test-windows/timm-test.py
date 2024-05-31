'''
Author: wangwei83 wangwei83@cuit.edu.cn
Date: 2024-05-31 21:42:00
LastEditors: wangwei83 wangwei83@cuit.edu.cn
LastEditTime: 2024-05-31 21:53:13
FilePath: /wangwei/X-23d-Y-ai-Z-detection/misc-test-windows/timm-test.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
# 可能服务器上vpn有点问题
import timm

# 指定模型名称
model_name = 'resnet50'

# 创建模型
model = timm.create_model(model_name, pretrained=True)

# 打印模型结构
print(model)