'''
Author: wangwei83 wangwei83@cuit.edu.cn
Date: 2024-05-27 19:06:20
LastEditors: wangwei83 wangwei83@cuit.edu.cn
LastEditTime: 2024-05-27 19:53:15
FilePath: /wangwei/X-23d-Y-ai-Z-detection/M3DM-from-Scratch/main.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import argparse
import os

from dataset import mvtec3d_classes  # Import the module that contains the 'mvtec3d_classes' function

def run_3d_ads(args):
    # Add your implementation here
    print("Run 3D-ADS method")
    
    if args.dataset_type =='mvtec3d':
        classes = mvtec3d_classes()
    
    METHOD_NAME = [args.method_name]
    
   
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
   
    # 添加根据命令行参数传入的参数
    parser.add_argument('--method_name', type=str, default='DINO+Point_MAE', help='Method name')
    parser.add_argument('--memory_bank', type=str, default='multiple', help='Memory bank type')
    parser.add_argument('--rgb_backbone_name', type=str, default='vit_base_patch8_224_dino', help='RGB backbone name')
    parser.add_argument('--xyz_backbone_name', type=str, default='Point_MAE', help='XYZ backbone name')
    parser.add_argument('--save_feature', action='store_true', help='Save feature flag')
    parser.add_argument('--dataset_type', type=str, default='mvtec3d', help='Type of the dataset')
    # 解释器解释命令行参数
    args = parser.parse_args()

    print(args.method_name)
    print(args.memory_bank)
    print(args.rgb_backbone_name)
    print(args.xyz_backbone_name)
    print(args.save_feature)

    # current_path = os.getcwd()
    # print("Current Path:", current_path)

    run_3d_ads(args)
    