'''
Author: wangwei83 wangwei83@cuit.edu.cn
Date: 2024-05-28 10:48:48
LastEditors: wangwei83 wangwei83@cuit.edu.cn
LastEditTime: 2024-05-28 10:56:06
FilePath: /wangwei/X-23d-Y-ai-Z-detection/M3DM-from-Scratch/utils/preprocessing.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''

import argparse
from pathlib import Path


if __name__ == '__main__':    # Your code here
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_path', type=str, help='path to the dataset')
    args=parser.parse_args()
    root_path = args.dataset_path
    path=Path(root_path)
    print(path)