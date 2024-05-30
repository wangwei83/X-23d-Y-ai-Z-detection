'''
Author: wangwei83 wangwei83@cuit.edu.cn
Date: 2024-05-28 10:48:48
LastEditors: wangwei83 wangwei83@cuit.edu.cn
LastEditTime: 2024-05-30 09:05:55
FilePath: /wangwei/X-23d-Y-ai-Z-detection/M3DM-from-Scratch/utils/preprocessing.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''

import argparse
import math
import os
from pathlib import Path
from PIL import Image
import numpy as np
import mvtec3d_util as mvt_util
import open3d as o3d

def get_edges_of_pc(organized_pc):
    # print(organized_pc)
    # 新的数组的第一维的大小是原数组的第一维和第二维的元素总数，第二维的大小是原数组的第三维的大小。
    unorganized_edges_pc=organized_pc[0:10,:,:].reshape(organized_pc[0:10,:,:].shape[0]*organized_pc[0:10,:,:].shape[1],organized_pc[0:10,:,:].shape[2])
    # print(unorganized_edges_pc)
    # 沿着行的方向进行拼接
    unorganized_edges_pc=np.concatenate((unorganized_edges_pc,organized_pc[-10:,:,:].reshape(organized_pc[-10:,:,:].shape[0]*organized_pc[-10:,:,:].shape[1],organized_pc[-10:,:,:].shape[2])),axis=0)
    unorganized_edges_pc = np.concatenate((unorganized_edges_pc,organized_pc[:,0:10,:].reshape(organized_pc[:,0:10,:].shape[0]*organized_pc[:,0:10,:].shape[1],organized_pc[:,0:10,:].shape[2])),axis=0)
    unorganized_edges_pc = np.concatenate((unorganized_edges_pc,organized_pc[:,-10:,:].reshape(organized_pc[:,-10:,:].shape[0]*organized_pc[:,-10:,:].shape[1],organized_pc[:,-10:,:].shape[2])),axis=0)
    return unorganized_edges_pc
    
def get_plane_eq(unorganized_pc,ransac_n_pts=50):
    # print(unorganized_pc)
    o3d_pc=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(unorganized_pc))
    # print('o3d_pc')
    # print(o3d_pc)
    plane_model, inliers = o3d_pc.segment_plane(distance_threshold=0.04, ransac_n=ransac_n_pts, num_iterations=1000)
    return plane_model
    
def remove_plane(organized_pc_clean, organized_rgb,distance_threshold=0.005):
    # Remove the plane
    
    unorganized_pc  = mvt_util.organized_pc_to_unorganized_pc(organized_pc_clean)
    unorganized_rgb = mvt_util.organized_pc_to_unorganized_pc(organized_rgb)
    clean_planeless_unorganized_pc =unorganized_pc.copy()
    planeless_unorganized_rgb = unorganized_rgb.copy()
    
    # REMOVE PLANE
    plance_model = get_plane_eq(get_edges_of_pc(organized_pc_clean))
    distance=np.abs(np.dot(np.array(plance_model),np.hstack((clean_planeless_unorganized_pc,np.ones((clean_planeless_unorganized_pc.shape[0],1)))).T))
    plane_indices=np.argwhere(distance<distance_threshold)
    # print(plane_indices)
    
    planeless_unorganized_rgb[plane_indices] = 0
    clean_planeless_unorganized_pc[plane_indices] = 0
    clean_planeless_organized_pc = clean_planeless_unorganized_pc.reshape(organized_pc_clean.shape[0],
                                                                        organized_pc_clean.shape[1],
                                                                        organized_pc_clean.shape[2])
    planeless_organized_rgb = planeless_unorganized_rgb.reshape(organized_rgb.shape[0],
                                                                organized_rgb.shape[1],
                                                                organized_rgb.shape[2])
    
    return clean_planeless_organized_pc, planeless_organized_rgb

def roundup_next_100(x):
    # 将输入的x向上取整到最近的100的倍数
    return int(math.ceil(x/100.0))*100

def connected_components_cleaning(organized_pc,organized_rgb,image_path):    
    unorganized_pc = mvt_util.organized_pc_to_unorganized_pc(organized_pc)
    unorganized_rgb = mvt_util.organized_pc_to_unorganized_pc(organized_rgb)
    
    nonzero_indices = np.nonzero(np.all(unorganized_pc!=0,axis=1))[0]
    unorganized_pc_no_zeros = unorganized_pc[nonzero_indices,:]
    # 采用Open3d库的Pointcloud类将无组织的点云数据转换为Open3d的点云对象
    o3d_pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(unorganized_pc_no_zeros))
    labels = np.array(o3d_pc.cluster_dbscan(eps=0.006, min_points=30, print_progress=False))
    # print(f"Found {len(np.unique(labels))} clusters in {image_path}")
    
def pad_cropped_pc(cropped_pc, single_channel=False):
    orig_h,orig_w = cropped_pc.shape[0],cropped_pc.shape[1]
    round_orig_h=roundup_next_100(orig_h)
    round_orig_w = roundup_next_100(orig_w)
    large_side = max(round_orig_h,round_orig_w)
    
    a=(large_side-orig_h)//2
    aa = large_side-orig_h-a
    
    b=(large_side-orig_w)//2
    bb = large_side-orig_w-b
    
    if single_channel:
        return np.pad(cropped_pc,pad_width=((a,aa),(b,bb)),mode='constant')
    else:
        return np.pad(cropped_pc,pad_width=((a,aa),(b,bb),(0,0)),mode='constant') 

def preprocess_pc(tiff_path):
    organized_pc = mvt_util.read_tiff_organized_pc(tiff_path)
    rgb_path = str(tiff_path).replace("xyz", "rgb").replace("tiff", "png")
    print(rgb_path)
    gt_path = str(tiff_path).replace("xyz", "gt").replace("tiff", "png")
    # print(gt_path)
    organized_rgb = np.array(Image.open(rgb_path))
    organized_gt=None
    gt_exists = os.path.isfile(gt_path)
    if gt_exists:
        organized_gt = np.array(Image.open(gt_path))
        
    # REMOVE PLANE，去除平面的规则点云和RGB图像
    planeless_organized_pc, planeless_organized_rgb = remove_plane(organized_pc, organized_rgb)

    # PAD WITH ZEROS TO LARGEST SIDE
    # (so that the final image is square)
    padded_planeless_organized_pc = pad_cropped_pc(planeless_organized_pc,single_channel=False)
    padded_planeless_organized_rgb = pad_cropped_pc(planeless_organized_rgb,single_channel=False)
    if gt_exists:
        padded_planeless_organized_gt = pad_cropped_pc(organized_gt,single_channel=True)
    # 点云连通组件分析和清理工作，跟聚类是不是有什么关系
    # organized_clustered_pc,organized_clustered_rgb=connected_components_cleaning(padded_planeless_organized_pc,padded_planeless_organized_rgb,tiff_path)

def tiff_to_pointcloud(path):
    
    # 获取文件名（不包括扩展名）
    base_name = os.path.splitext(os.path.basename(path))[0]
    # 创建输出文件的路径
    output_path = base_name + '.txt'

    # 读取tiff文件
    rgb_path = str(path).replace("xyz", "rgb").replace("tiff", "png")  
    organized_rgb = np.array(Image.open(rgb_path))
    
    # 将3D数组转换为2D数组
    organized_rgb_2d = organized_rgb.reshape(-1, organized_rgb.shape[-1])
    # 保存2D数组
    np.savetxt("../../"+output_path, organized_rgb_2d, fmt='%d')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_path', type=str, help='path to the dataset')
    args=parser.parse_args()
    root_path = args.dataset_path
    paths=Path(root_path).rglob('*.tiff')
    print(f"Found {len(list(paths))} tiff files in {root_path}")

    # 真实代码分支
    # for path in Path(root_path).rglob('*.tiff'):
    #     preprocess_pc(path
    
    # 测试分支，减少文件的读取次数
    for i, path in enumerate(Path(root_path).rglob('*.tiff')):
        # print(i)
        # print(path)
        # # 这个函数的主要功能是将 TIFF 图像文件转换为点云数据
        # tiff_to_pointcloud(path)

        preprocess_pc(path)
        if i >= 5:
            break
        