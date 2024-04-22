import open3d as o3d
import numpy as np

# 生成一些随机的3D点云数据
points = np.random.rand(1000, 3)  # 1000个随机3D点

# 将NumPy数组转换为Open3D的点云格式
point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(points)

# 可视化点云
o3d.visualization.draw_geometries([point_cloud])
