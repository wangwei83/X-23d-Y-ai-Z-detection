import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering, KMeans
from sklearn.datasets import make_moons

# 生成非线性可分的数据
X, y = make_moons(n_samples=200, noise=0.1, random_state=0)

# 谱聚类
def spectral_clustering(X, n_clusters):
    # 实例化谱聚类
    sc = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors',
                            assign_labels='kmeans')
    labels = sc.fit_predict(X)
    return labels

# 运行谱聚类
labels = spectral_clustering(X, n_clusters=2)

# 可视化结果
plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', edgecolor='k', s=50)
plt.title('Spectral Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.colorbar()
plt.show()
