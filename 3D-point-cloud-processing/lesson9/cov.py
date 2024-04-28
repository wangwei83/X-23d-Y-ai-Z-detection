import numpy as np

# 创建数据矩阵
data = np.array([
    [1, 2],
    [3, 6],
    [5, 10]
])

# 计算协方差矩阵
cov_matrix = np.cov(data, rowvar=False)  # rowvar=False 指定数据的每一列是一个变量

# 打印协方差矩阵
print("协方差矩阵:")
print(cov_matrix)
