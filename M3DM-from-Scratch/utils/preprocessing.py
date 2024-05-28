'''
Author: wangwei83 wangwei83@cuit.edu.cn
Date: 2024-05-28 10:48:48
LastEditors: wangwei83 wangwei83@cuit.edu.cn
LastEditTime: 2024-05-28 11:28:55
FilePath: /wangwei/X-23d-Y-ai-Z-detection/M3DM-from-Scratch/utils/preprocessing.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''

import argparse
from pathlib import Path
from PIL import Image
import numpy as np
import mvtec3d_util as mvt_util

def preprocess_pc(tiff_path):
    organized_pc = mvt_util.read_tiff_organized_pc(tiff_path)
    # print(f"Preprocessing {tiff_path}")
    # print(f"Shape of the organized point cloud: {organized_pc.shape}")
    # print(f"Data type of the organized point cloud: {organized_pc.dtype}")
    rgb_path = str(tiff_path).replace("xyz", "rgb").replace("tiff", "png")
    gt_path = str(tiff_path).replace("xyz", "gt").replace("tiff", "png")
    # print(f"Saving RGB image to {rgb_path}")
    # print(f"Saving ground truth image to {gt_path}")
    organized_rgb=np.array(Image.open(rgb_path))
    print(f"Shape of the organized RGB image: {organized_rgb.shape}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_path', type=str, help='path to the dataset')
    args=parser.parse_args()
    root_path = args.dataset_path
    paths=Path(root_path).rglob('*.tiff')
    print(f"Found {len(list(paths))} tiff files in {root_path}")
    
    for path in Path(root_path).rglob('*.tiff'):
        preprocess_pc(path)