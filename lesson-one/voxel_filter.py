# 实现voxel滤波，并加载数据集中的文件进行验证

import open3d as o3d 
import os
import numpy as np
from pyntcloud import PyntCloud

# 功能：对点云进行voxel滤波
# 输入：
#     point_cloud：输入点云
#     leaf_size: voxel尺寸
def voxel_filter(point_cloud, leaf_size):
    filtered_points = []
    # 作业3
    # 屏蔽开始
    d_max = np.max(point_cloud,axis=0)
    d_min = np.min(point_cloud,axis=0)
    d = (d_max-d_min)/leaf_size

    h_x = (point_cloud.iloc[:,0]-d_min[0])/leaf_size
    h_y = (point_cloud.iloc[:,1]-d_min[1])/leaf_size
    h_z = (point_cloud.iloc[:,2]-d_min[2])/leaf_size

    h = np.floor(h_x+h_y*d[0]+h_z*d[0]*d[1])  #计算点云对应index

    point_cloud = np.hstack([point_cloud,np.expand_dims(h,axis=1)])
    point_cloud = point_cloud[np.argsort(point_cloud[1,3])]

    n = point_cloud.shape[0]
    index = []
    for i in range(n):
        if i < n-1 and point_cloud[i,3] == point_cloud[i+1,3] :
            index.append[i]
        else:
            index.append[i]

            if mode == "mean":
                point = np.mean(point_cloud[index,:3],axis=0)
                filtered_points.append[point]
            elif mode == "random":
                id = random.choice(index)
                filtered_points.append(point_cloud[id,:3])
            index =[]
    # 屏蔽结束

    # 把点云格式改成array，并对外返回
    filtered_points = np.array(filtered_points, dtype=np.float64)
    return filtered_points

def main():
    # # 从ModelNet数据集文件夹中自动索引路径，加载点云
    # cat_index = 10 # 物体编号，范围是0-39，即对应数据集中40个物体
    # root_dir = '/Users/renqian/cloud_lesson/ModelNet40/ply_data_points' # 数据集路径
    # cat = os.listdir(root_dir)
    # filename = os.path.join(root_dir, cat[cat_index],'train', cat[cat_index]+'_0001.ply') # 默认使用第一个点云
    # point_cloud_pynt = PyntCloud.from_file(file_name)

    data_dir='.\\lesson-one\\'

    with open(data_dir+'modelnet40_shape_names.txt') as f:
        a = f.readlines()
    for i in a:
        point_cloud_pynt = PyntCloud.from_file(
            data_dir+'{}/{}_0001.txt'.format(i.strip(), i.strip()), sep=",",
            names=["x", "y", "z", "nx", "ny", "nz"])
    # 转成open3d能识别的格式
    point_cloud_o3d = point_cloud_pynt.to_instance("open3d", mesh=False)
    #o3d.visualization.draw_geometries([point_cloud_o3d]) # 显示原始点云

    # 调用voxel滤波函数，实现滤波
    filtered_cloud = voxel_filter(point_cloud_pynt.points, 100.0)
    #point_cloud_o3d.points = o3d.utility.Vector3dVector(filtered_cloud)
    # 显示滤波后的点云
    #o3d.visualization.draw_geometries([point_cloud_o3d])

if __name__ == '__main__':
    main()
