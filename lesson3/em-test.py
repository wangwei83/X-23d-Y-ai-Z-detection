import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

def initialize_parameters(data, num_components):
    """初始化参数：均值、协方差和混合权重"""
    indices = np.random.choice(len(data), num_components, replace=False)
    means = data[indices]
    variances = np.array([np.cov(data, rowvar=False)] * num_components)
    weights = np.ones(num_components) / num_components
    return means, variances, weights

def e_step(data, means, variances, weights):
    """E步骤：计算后验概率，即每个点属于各高斯分量的概率"""
    num_samples = data.shape[0]
    num_components = means.shape[0]
    responsibilities = np.zeros((num_samples, num_components))
    for i in range(num_components):
        diff = data - means[i]
        inv_var = np.linalg.inv(variances[i])
        term = np.exp(-0.5 * np.sum(diff @ inv_var * diff, axis=1))
        coef = (2 * np.pi)**(-data.shape[1]/2) * np.linalg.det(variances[i])**(-0.5)
        responsibilities[:, i] = weights[i] * coef * term
    sum_responsibilities = responsibilities.sum(axis=1, keepdims=True)
    responsibilities /= sum_responsibilities
    return responsibilities

def m_step(data, responsibilities):
    """M步骤：更新参数（均值、协方差、权重）"""
    num_components = responsibilities.shape[1]
    weights = responsibilities.sum(axis=0) / responsibilities.sum()
    means = np.dot(responsibilities.T, data) / responsibilities.sum(axis=0)[:, np.newaxis]
    variances = np.zeros((num_components, data.shape[1], data.shape[1]))
    for i in range(num_components):
        diff = data - means[i]
        weighted_diff = responsibilities[:, i, np.newaxis] * diff
        variances[i] = np.dot(weighted_diff.T, diff) / responsibilities[:, i].sum()
    return means, variances, weights

def em_algorithm(data, num_components, num_iterations):
    """EM算法主循环"""
    means, variances, weights = initialize_parameters(data, num_components)
    for _ in range(num_iterations):
        responsibilities = e_step(data, means, variances, weights)
        means, variances, weights = m_step(data, responsibilities)
    return means, variances, weights

# 示例数据生成
np.random.seed(0)
data = np.random.randn(300, 2) * 2  # 生成更大范围的数据以便观察

# 设定参数和迭代次数
num_components = 4
num_iterations = 100

# 运行EM算法
means, variances, weights = em_algorithm(data, num_components, num_iterations)

# 可视化结果
plt.figure(figsize=(10, 8))
plt.scatter(data[:, 0], data[:, 1], s=30, color='red', label='Data points')
x, y = np.linspace(-10, 10, 100), np.linspace(-10, 10, 100)
X, Y = np.meshgrid(x, y)
XX = np.array([X.ravel(), Y.ravel()]).T
for mean, variance, weight in zip(means, variances, weights):
    Z = np.exp(-0.5 * (XX - mean).dot(np.linalg.inv(variance)).dot((XX - mean).T).diagonal())
    Z = Z.reshape(X.shape)
    plt.contour(X, Y, Z, levels=14, norm=LogNorm(vmin=1.0, vmax=1000.0), linewidths=2)

plt.title('Visualization of GMM Clustering')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.legend()
plt.grid(True)
plt.show()
