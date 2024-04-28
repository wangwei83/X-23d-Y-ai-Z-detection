import numpy as np

def find_closest_points(p, q):
    # 寻找最近点对
    indices = []
    for point in p:
        distances = np.linalg.norm(q - point, axis=1)
        closest_point_index = np.argmin(distances)
        indices.append(closest_point_index)
    return q[indices]

def best_fit_transform(p, q):
    # 计算最佳变换
    p_centroid = np.mean(p, axis=0)
    q_centroid = np.mean(q, axis=0)
    p_centered = p - p_centroid
    q_centered = q - q_centroid
    H = np.dot(p_centered.T, q_centered)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)
    if np.linalg.det(R) < 0:
       Vt[-1, :] *= -1
       R = np.dot(Vt.T, U.T)
    t = q_centroid - np.dot(R, p_centroid)
    return R, t

def icp(a, b, iterations=10):
    p = np.copy(a)
    for i in range(iterations):
        q = find_closest_points(p, b)
        R, t = best_fit_transform(p, q)
        p = np.dot(p, R.T) + t
        if np.linalg.norm(t) < 0.001 and np.linalg.norm(R - np.eye(2)) < 0.001:
            break
    return R, t



# 示例点集
a = np.array([[0, 0], [1, 0], [0, 1]])
b = np.array([[1, 1], [2, 1], [1, 2]])

# 运行 ICP
R, t = icp(a, b)
print('Rotation:', R)
print('Translation:', t)
