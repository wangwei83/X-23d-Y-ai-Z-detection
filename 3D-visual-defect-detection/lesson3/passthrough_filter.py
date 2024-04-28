import open3d as o3d
import numpy as np

def passthrough_filter(point_cloud, axis, min_val, max_val):
    # 创建一个选择性索引，保留在指定轴上处于[min_val, max_val]范围内的点
    points = np.asarray(point_cloud.points)
    if axis == 'x':
        mask = (points[:, 0] >= min_val) & (points[:, 0] <= max_val)
    elif axis == 'y':
        mask = (points[:, 1] >= min_val) & (points[:, 1] <= max_val)
    elif axis == 'z':
        mask = (points[:, 2] >= min_val) & (points[:, 2] <= max_val)
    else:
        raise ValueError("Axis must be 'x', 'y', or 'z'")

    # 应用掩码
    filtered_points = points[mask]
    filtered_cloud = o3d.geometry.PointCloud()
    filtered_cloud.points = o3d.utility.Vector3dVector(filtered_points)
    return filtered_cloud

# 加载点云数据
pcd = o3d.io.read_point_cloud(r"C:\Users\19002\Desktop\cloud_lesson\3D-visual-defect-detection\lesson3\welding_scene.ply")

# 应用直通滤波，例如在Z轴上，保留从0.0到1.0的点
filtered_pcd = passthrough_filter(pcd, 'z', 0.0, 1000.0)

# 可视化过滤后的点云
o3d.visualization.draw_geometries([filtered_pcd])
