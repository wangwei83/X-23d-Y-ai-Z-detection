
'''
Author: wangwei83 wangwei83@cuit.edu.cn
Date: 2024-05-28 11:05:44
LastEditors: wangwei83 wangwei83@cuit.edu.cn
LastEditTime: 2024-05-28 23:27:38
FilePath: /wangwei/X-23d-Y-ai-Z-detection/M3DM-from-Scratch/utils/mvtec3d_util.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''

import tifffile as tiff

def organized_pc_to_unorganized_pc(organized_pc):
    '''
    Convert organized point cloud to unorganized point cloud
    '''
    # 将organized_pc数组重塑为一个二维数组，新数组的第一维的大小是原数组的第一维和第二维的元素总数，第二维的大小是原数组的第三维的大小。
    # 这种操作通常用于将一个多维数组“展平”为一个二维数组，例如在机器学习中，我们可能需要将一个三维的图像数据集展平为二维，以便可以将其输入到一个机器学习模型中。
    return organized_pc.reshape(organized_pc.shape[0] * organized_pc.shape[1], organized_pc.shape[2])

def read_tiff_organized_pc(tiff_path):
    '''
    Read a tiff file and return the organized point cloud
    '''
    organized_pc = tiff.imread(tiff_path)
    return organized_pc