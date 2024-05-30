
'''
Author: wangwei83 wangwei83@cuit.edu.cn
Date: 2024-05-28 11:05:44
LastEditors: wangwei83 wangwei83@cuit.edu.cn
LastEditTime: 2024-05-30 19:39:46
FilePath: /wangwei/X-23d-Y-ai-Z-detection/M3DM-from-Scratch/utils/mvtec3d_util.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''

import tifffile as tiff
import torch

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
def resize_organized_pc(organized_pc, target_height=224, target_width=224,tensor_out=True):
    # 这行代码将 organized_pc 转换为一个 PyTorch 张量，并对其进行了一系列的变换，包括重新排列维度、添加新的维度和确保内存布局连续
    torch_organized_pc = torch.tensor(organized_pc).permute(2,0,1).unsqueeze(dim=0).contiguous()
    
    # 这行代码使用 PyTorch 的 interpolate 函数对一个名为 torch_organized_pc 的张量进行了尺寸调整（或称为插值）
    torch_resized_organized_pc = torch.nn.functional.interpolate(torch_organized_pc, size=(target_height, target_width),
                                                                 mode='nearest')
    if tensor_out:
        return torch_resized_organized_pc.squeeze(dim=0).contiguous()
    else:
        return torch_resized_organized_pc.squeeze().permute(1,2,0).contiguous().numpy()
    
    # return organized_pc[::organized_pc.shape[0]//target_height, ::organized_pc.shape[1]//target_width]
    
def organized_pc_to_depth_map(organized_pc):
    '''
    Convert organized point cloud to depth map
    '''
    return organized_pc[:,:,2]