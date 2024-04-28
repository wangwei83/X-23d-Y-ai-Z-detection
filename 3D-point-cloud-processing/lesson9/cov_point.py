import numpy as np
from scipy.linalg import svd

def compute_centroid(point_cloud):
    """计算点云的质心"""
    return np.mean(point_cloud, axis=0)

def compute_covariance_matrix(A, B):
    """计算两个点云之间的协方差矩阵"""
    A_mean = compute_centroid(A)
    B_mean = compute_centroid(B)
    A_centered = A - A_mean
    B_centered = B - B_mean
    covariance_matrix = np.dot(A_centered.T, B_centered) / (len(A) - 1)
    return covariance_matrix

def find_optimal_rotation(covariance_matrix):
    """使用 SVD 找到最佳旋转矩阵"""
    U, _, Vt = svd(covariance_matrix)
    R = np.dot(Vt.T, U.T)
    # 确保得到一个正确的旋转矩阵（行列式为 1）
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = np.dot(Vt.T, U.T)
    return R

# 示例点云数据
A = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])

B = np.array([
    [2, 3, 4],
    [5, 6, 7],
    [8, 9, 10]
])

# 计算协方差矩阵
covariance_matrix = compute_covariance_matrix(A, B)
print("协方差矩阵:")
print(covariance_matrix)

# 找到最佳旋转矩阵
R = find_optimal_rotation(covariance_matrix)
print("最佳旋转矩阵:")
print(R)
