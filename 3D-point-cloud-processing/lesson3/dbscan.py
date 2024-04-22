import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons

# 生成数据
X, _ = make_moons(n_samples=300, noise=0.1, random_state=42)

# 初始化DBSCAN对象
# eps是邻域半径，min_samples是形成密集区所需的最小样本数
dbscan = DBSCAN(eps=0.2, min_samples=5)

# 执行DBSCAN聚类
labels = dbscan.fit_predict(X)

# 可视化聚类结果
plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', edgecolors='k')
plt.title('DBSCAN Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.colorbar()
plt.show()
