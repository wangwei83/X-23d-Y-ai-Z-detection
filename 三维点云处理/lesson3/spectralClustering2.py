import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering
from sklearn.datasets import make_circles

# 生成数据
X, _ = make_circles(n_samples=300, factor=.5, noise=.05)

# 创建谱聚类对象，设置聚类数为2
clustering = SpectralClustering(n_clusters=3, affinity='nearest_neighbors', n_neighbors=10, random_state=0)

# 拟合模型
labels = clustering.fit_predict(X)

# 可视化结果
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', edgecolors='k')
plt.title("Spectral Clustering")
plt.show()
