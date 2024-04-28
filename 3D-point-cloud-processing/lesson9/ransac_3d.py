import numpy as np
import open3d as o3d

def random_three_points(data):
    """随机选择三个点以拟合一个平面"""
    indices = np.random.choice(data.shape[0], 3, replace=False)
    return data[indices, :]

def plane_from_points(points):
    """从三个点计算平面方程"""
    p1, p2, p3 = points
    # 计算两个向量
    v1 = p2 - p1
    v2 = p3 - p1
    # 向量叉乘得到法向量
    normal = np.cross(v1, v2)
    normal = normal / np.linalg.norm(normal)  # 单位化法向量
    # 平面方程: Ax + By + Cz + D = 0
    D = -np.dot(normal, p1)
    return normal, D

def compute_inliers(data, normal, D, threshold):
    """计算平面内点"""
    distances = np.abs(np.dot(data, normal) + D)
    inlier_mask = distances < threshold
    return inlier_mask

def ransac_point_cloud(data, num_iterations, threshold):
    best_inliers = []
    for _ in range(num_iterations):
        maybe_points = random_three_points(data)
        maybe_normal, maybe_D = plane_from_points(maybe_points)
        maybe_inliers = compute_inliers(data, maybe_normal, maybe_D, threshold)
        
        if len(maybe_inliers) > len(best_inliers):
            best_inliers = maybe_inliers
            best_normal = maybe_normal
            best_D = maybe_D

    print(f"Found plane with normal {best_normal} and D {best_D}")
    return data[best_inliers], best_normal, best_D

# 加载点云数据
pcd = o3d.io.read_point_cloud(r"C:\Users\19002\Desktop\cloud_lesson\3D-point-cloud-processing\lesson9\bun000.ply")  # 请将路径替换为你的点云文件

# 使用Open3D可视化点云
o3d.visualization.draw_geometries([pcd], window_name="PLY Visualization", width=800, height=600)

points = np.asarray(pcd.points)

# 运行RANSAC
inlier_points, normal, D = ransac_point_cloud(points, 1000, 0.01)
inlier_cloud = o3d.geometry.PointCloud()
inlier_cloud.points = o3d.utility.Vector3dVector(inlier_points)

# 可视化结果
o3d.visualization.draw_geometries([inlier_cloud], window_name="Inlier Points")
