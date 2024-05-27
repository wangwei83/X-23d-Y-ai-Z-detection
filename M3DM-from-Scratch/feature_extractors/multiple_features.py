'''
Author: wangwei83 wangwei83@cuit.edu.cn
Date: 2024-05-27 21:34:17
LastEditors: wangwei83 wangwei83@cuit.edu.cn
LastEditTime: 2024-05-27 21:49:35
FilePath: /wangwei/X-23d-Y-ai-Z-detection/M3DM-from-Scratch/feature_extractors/multiple_features.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
from feature_extractors.features import Features

import numpy as np

class DoubleRGBPointFeatures(Features):
    def add_sample_to_mem_bank(self, sample,class_name=None):
        pass
    
    # def predict(self, sample,mask,label):
    #     pass
    