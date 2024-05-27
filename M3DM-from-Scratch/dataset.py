'''
Author: wangwei83 wangwei83@cuit.edu.cn
Date: 2024-05-27 19:43:50
LastEditors: wangwei83 wangwei83@cuit.edu.cn
LastEditTime: 2024-05-27 22:44:06
FilePath: /wangwei/X-23d-Y-ai-Z-detection/M3DM-from-Scratch/dataset.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''

import glob
import os
import numpy as np
from torch.utils.data import DataLoader

def mvtec3d_classes():
    return ['bagel', 
            'cable_gland', 
            'carrot', 
            'cookie', 
            'dowel', 
            'foam', 
            'peach', 
            'potato', 
            'rope', 
            'tire']
class BaseAnomalyDetectionDataset:
    def __init__(self, split, class_name, img_size, dataset_path='datasets/eyecandies_preprocessed'):
        self.cls = class_name
        self.img_size = img_size
        self.img_path = os.path.join(dataset_path, self.cls, split)
        
        
class TrainDataset(BaseAnomalyDetectionDataset):
    def __init__(self, class_name, img_size, dataset_path='datasets/eyecandies_preprocessed'):
        
        super().__init__(split='train', class_name=class_name, img_size=img_size, dataset_path=dataset_path)
        self.img_paths,self.labels = self.load_dataset()
        
    
    def __len__(self):
        return len(self.img_paths)
    
    def load_dataset(self):
        img_paths = []  # Change variable name from "img_tot_paths" to "img_paths"
        labels = []  # Change variable name from "tot_labels" to "labels"
        rgb_paths = glob.glob(os.path.join(self.img_path, 'good','rgb')+"/*.png")
        tiff_paths = glob.glob(os.path.join(self.img_path, 'good','xyz')+"/*.tiff")
        for img_name in os.listdir(self.img_path):
            img_path = os.path.join(self.img_path, img_name)
            img_paths.append(img_path)
            labels.append(1)
            
        return img_paths, labels
       
        
def get_data_loader(split, class_name, img_size, args):
    if split in ['train']:
        dataset = TrainDataset(class_name=class_name, img_size=img_size, dataset_path=args.dataset_path)
    
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1, drop_last=False, pin_memory=True)
    return data_loader