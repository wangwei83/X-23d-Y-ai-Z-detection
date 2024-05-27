
'''
Author: wangwei83 wangwei83@cuit.edu.cn
Date: 2024-05-27 21:03:16
LastEditors: wangwei83 wangwei83@cuit.edu.cn
LastEditTime: 2024-05-27 22:35:20
FilePath: /wangwei/X-23d-Y-ai-Z-detection/M3DM-from-Scratch/m3dm_runner.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
from feature_extractors import multiple_features

from dataset import get_data_loader  # Import the get_data_loader function
from tqdm import tqdm  # Import the tqdm package

class M3DM():
    def __init__(self, args):
        self.args = args
        # print("M3DM model is initialized")
        self.image_size=args.img_size
        # print("Image size: ", self.image_size)
        self.count = args.max_sample
        # print("Count: ", self.count)
        
        if args.method_name == 'DINO+Point_MAE':
            self.method_name = {'DINO+Point_MAE': multiple_features.DoubleRGBPointFeatures(args),}
        
    

    def fit(self, class_name):
        # print("Fitting the model for the class: ", cls)
        train_loader = get_data_loader("train", class_name=class_name, img_size=self.image_size, args=self.args)
        
        flag =0
        for sample, _ in tqdm(train_loader, desc='Processing samples'):
            # if sample == self.count:  # Replace "i" with "sample"
            #     break
            # print("Image shape: ", img.shape)
            # print("Mask shape: ", mask.shape)
            
            # self.method_name['DINO+Point_MAE'].add_sample_to_mem_bank(img, class_name)
            print(self.methods.values())
            for method in self.methods.values():
                method.add_sample_to_mem_bank(sample, class_name)
            flag = 1