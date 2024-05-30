'''
Author: wangwei83 wangwei83@cuit.edu.cn
Date: 2024-05-27 19:43:50
LastEditors: wangwei83 wangwei83@cuit.edu.cn
LastEditTime: 2024-05-30 19:41:16
FilePath: /wangwei/X-23d-Y-ai-Z-detection/M3DM-from-Scratch/dataset.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''

import glob
import os
import numpy as np
from torch.utils.data import DataLoader
from PIL import Image
from utils.mvtec3d_util import *
from torchvision import transforms

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
RGB_SIZE = 224

class BaseAnomalyDetectionDataset:
    def __init__(self, split, class_name, img_size, dataset_path='datasets/eyecandies_preprocessed'):
        # self.cls = class_name
        # self.img_size = img_size
        # self.img_path = os.path.join(dataset_path, self.cls, split)
        self.IMAGENET_MEAN = [0.485, 0.456, 0.406]
        self.IMAGENET_STD = [0.229, 0.224, 0.225]
        self.cls=   class_name
        self.size = img_size
        self.img_path = os.path.join(dataset_path, self.cls, split)
        self.rgb_transform = transforms.Compose([
            transforms.Resize((RGB_SIZE, RGB_SIZE),interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD)
        ])        
        
class TrainDataset(BaseAnomalyDetectionDataset):
    def __init__(self, class_name, img_size, dataset_path='datasets/eyecandies_preprocessed'):
        
        super().__init__(split='train', class_name=class_name, img_size=img_size, dataset_path=dataset_path)
        self.img_paths,self.labels = self.load_dataset()
        
    def load_dataset(self):
        img_tot_paths = []  
        tot_labels = []  
        rgb_paths = glob.glob(os.path.join(self.img_path, 'good','rgb')+"/*.png")
        tiff_paths = glob.glob(os.path.join(self.img_path, 'good','xyz')+"/*.tiff")
        rgb_paths.sort()
        tiff_paths.sort()
        # 配对
        sample_paths = list(zip(rgb_paths, tiff_paths))
        img_tot_paths.extend(sample_paths)  
        tot_labels.extend([0]*len(sample_paths))
            
        return img_tot_paths, tot_labels
    
    def __len__(self):
        return len(self.img_paths)
      
    def __getitem__(self, idx):
        img_path,label=self.img_paths[idx],self.labels[idx]
        rgb_path = img_path[0]
        tiff_path = img_path[1]
        img = Image.open(rgb_path).convert('RGB')
        
        img=self.rgb_transform(img)
        organized_pc = read_tiff_organized_pc(tiff_path)

        depth_map_3channel = np.repeat(organized_pc_to_depth_map(organized_pc)[:,:,np.newaxis], 3, axis=2)     
        resized_depth_map_3channel = resize_organized_pc(depth_map_3channel)
        resized_organized_pc = resize_organized_pc(organized_pc, target_height=self.size, target_width=self.size)
        resized_organized_pc =resized_organized_pc.clone().detach().float()    
                 
        return (img,resized_organized_pc,resized_depth_map_3channel), label
        
    
    
    
       
        
def get_data_loader(split, class_name, img_size, args):
    if split in ['train']:
        dataset = TrainDataset(class_name=class_name, img_size=img_size, dataset_path=args.dataset_path)
    
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1, drop_last=False, pin_memory=True)
    return data_loader