from sklearn.cluster import MeanShift
import numpy as np
import matplotlib.pyplot as plt

# 生成示例数据
X = np.random.randn(300, 2)  # 生成300个2维正态分布的点
X = np.vstack([X + [3, 3], X + [-3, -3], X + [3, -3]])  # 添加偏移创建明显的聚类

# 应用Mean Shift算法
meanshift = MeanShift(bandwidth=2)  # bandwidth决定了核的大小
meanshift.fit(X)

# 获取标签和聚类中心
labels = meanshift.labels_
cluster_centers = meanshift.cluster_centers_

# 可视化结果
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', marker='o')
plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='red', s=100, alpha=0.5)  # 聚类中心
plt.title('Mean Shift Clustering')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()