# 文件功能：
#     1. 从数据集中加载点云数据
#     2. 从点云数据中滤除地面点云
#     3. 从剩余的点云中提取聚类

import numpy as np
import os
import struct
from sklearn import cluster, datasets, mixture
from sklearn.neighbors import KDTree
import random
from itertools import cycle, islice
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 功能：从kitti的.bin格式点云文件中读取点云
# 输入：
#     path: 文件路径
# 输出：
#     点云数组
def read_velodyne_bin(path):
    '''
    :param path:
    :return: homography matrix of the point cloud, N*3
    '''
    pc_list = []
    with open(path, 'rb') as f:
        content = f.read()
        pc_iter = struct.iter_unpack('ffff', content)
        for idx, point in enumerate(pc_iter):
            pc_list.append([point[0], point[1], point[2]])
    return np.asarray(pc_list, dtype=np.float32)

# 功能：从点云文件中滤除地面点
# 输入：
#     data: 一帧完整点云
# 输出：
#     segmengted_cloud: 删除地面点之后的点云
def ground_segmentation(data):
    # 作业1
    # 屏蔽开始
    n = len(data)
    iter_num = 100
    sigma = 0.3
    p=0.99
    outlier_ratio =0.5
    best_idx = []
    best_inliner = (1-outlier_ratio)*n
    best_A, best_B,best_C, best_D = 0,0,0,0

    for i in range(iter_num):
        random_index = random.sample(range(n),3)
        point0 = data[random_index[0]]
        point1 = data[random_index[1]]
        point2 = data[random_index[2]]

        vector0_1 = point1-point0
        vector0_2 = point2-point0
        N = np.cross(vector0_1,vector0_2)
        A,B,C = N[0],N[1],N[2]
        D = -np.dot(N,point0)
        inliners =0
        distance = abs(np.dot(data,N)+D)/np.linalg.norm(N)
        idx = distance<sigma
        inliners = idx.sum()

        if inliners>best_inliner:
            best_idx =idx
            best_inliner = inliners
            best_A, best_B,best_C, best_D = A,B,C,D
        
        if inliners>(1-outlier_ratio)*n:
            break
    segmengted_cloud_idx = best_idx
    # 屏蔽结束

    print('origin data points num:', data.shape[0])
    print('segmented data points num:', segmengted_cloud_idx.shape[0])
    return segmengted_cloud_idx

# 功能：从点云中提取聚类
# 输入：
#     data: 点云（滤除地面后的点云）
# 输出：
#     clusters_index： 一维数组，存储的是点云中每个点所属的聚类编号（参考上一章内容容易理解）
def clustering(data):
    
    # 作业2
    # 屏蔽开始
    dis = 0.5
    min_sample =5
    n =len(data)

    leaf_size = 8
    if data.ndim == 1:
        data = data.reshape(-1, 1)
        
    kdtree = KDTree(data,leaf_size)

    core_set = set()
    unvisit_set = set(range(n))
    k = 0
    cluster_index = np.zeros(n,dtype=int)

    nearest_idx = kdtree.query_radius(data,dis)
    for i in range(n):
        if len(nearest_idx[i])>=min_sample:
            core_set.add(i)
    
    while len(core_set):
        unvisit_set_old = unvisit_set
        core = list(core_set)[np.random.randint(0,len(core_set))]
        unvisit_set = unvisit_set-set([core])
        visited = []
        visited.append(core)

        while len(visited):
            new_core = visited[0]
            if new_core in core_set:
                S = set(unvisit_set)&set(nearest_idx[new_core])
                visited.extend(list(S))
                unvisit_set = unvisit_set-S
            visited.remove(new_core)
        
        cluster = unvisit_set_old-unvisit_set
        core_set = core_set-cluster
        cluster_index[list(cluster)] = k
        k = k+1
        print("core_set:",len(core_set),"unvisit_set",len(unvisit_set))
    
    noise_cluster = unvisit_set
    cluster_index[list(noise_cluster)]=-1
    # 屏蔽结束

    return cluster_index

# 功能：显示聚类点云，每个聚类一种颜色
# 输入：
#      data：点云数据（滤除地面后的点云）
#      cluster_index：一维数组，存储的是点云中每个点所属的聚类编号（与上同）
def plot_clusters(data, cluster_index):
    ax = plt.figure().add_subplot(111, projection = '3d')
    colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                             '#f781bf', '#a65628', '#984ea3',
                                             '#999999', '#e41a1c', '#dede00']),
                                      int(max(cluster_index) + 1))))
    colors = np.append(colors, ["#000000"])
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], s=2, color=colors[cluster_index])
    plt.show()

def main():
    #root_dir = '.\lesson4\kitti' # 数据集路径  lesson2\kitti
    #root_dir = 'C:/Users/19002/Desktop/cloud_lesson/lesson2/kitti/' # 数据集路径  lesson2\kitti  C:\Users\19002\Desktop\cloud_lesson\lesson2\kitti
    #cat = os.listdir(root_dir)
    

    root_dir = '.\lesson2\kitti' # 数据集路径  lesson2\kitti
    cat = os.listdir(root_dir)
    iteration_num = len(cat)

    print("octree --------------")
    print(iteration_num)

    #cat = cat[1:]
    #iteration_num = len(cat)

    for i in range(iteration_num):
        filename = os.path.join(root_dir, cat[i])
        print('clustering pointcloud file:', filename)

        origin_points = read_velodyne_bin(filename)
        segmented_points = ground_segmentation(data=origin_points)
        cluster_index = clustering(segmented_points)

        plot_clusters(segmented_points, cluster_index)

if __name__ == '__main__':
    main()
