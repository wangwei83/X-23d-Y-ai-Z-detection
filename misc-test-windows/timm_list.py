

'''
Author: wangwei83 wangwei83@cuit.edu.cn
Date: 2024-06-02 21:58:06
LastEditors: wangwei83 wangwei83@cuit.edu.cn
LastEditTime: 2024-06-02 21:58:11
FilePath: /wangwei/X-23d-Y-ai-Z-detection/misc-test-windows/timm_list.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''

import timm

availableModels = timm.list_models(pretrained=True)
print(availableModels)
print(len(availableModels))
