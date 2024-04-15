
import matplotlib.pyplot as plt

import numpy as np
from scipy.stats import multivariate_normal

class GaussianMixtureModel:
    def __init__(self, n_components, n_iter, tol):
        self.n_components = n_components  # 高斯分布的数量
        self.n_iter = n_iter              # 最大迭代次数
        self.tol = tol                    # 收敛阈值
        self.weights = None               # 混合系数
        self.means = None                 # 均值
        self.covariances = None           # 协方差矩阵
    
    def fit(self, X):
        # 初始化参数
        n_samples, n_features = X.shape
        self.weights = np.ones(self.n_components) / self.n_components
        self.means = X[np.random.choice(n_samples, self.n_components, replace=False)]
        self.covariances = np.array([np.eye(n_features)] * self.n_components)
        
        log_likelihood_old = 0
        for i in range(self.n_iter):
            # E-step
            responsibilities = self._e_step(X)
            
            # M-step
            self._m_step(X, responsibilities)
            
            # 计算对数似然
            log_likelihood_new = self._log_likelihood(X)
            if np.abs(log_likelihood_new - log_likelihood_old) < self.tol:
                break
            log_likelihood_old = log_likelihood_new
    
    def _e_step(self, X):
        likelihood = np.array([multivariate_normal.pdf(X, mean=self.means[k], cov=self.covariances[k])
                               for k in range(self.n_components)]).T
        weighted_likelihood = likelihood * self.weights
        sum_likelihood = np.sum(weighted_likelihood, axis=1)[:, np.newaxis]
        return weighted_likelihood / sum_likelihood
    
    def _m_step(self, X, responsibilities):
        n_samples = X.shape[0]
        self.weights = np.mean(responsibilities, axis=0)
        self.means = np.dot(responsibilities.T, X) / np.sum(responsibilities, axis=0)[:, np.newaxis]
        #for k in range(self.n_components):
        #    X_centered = X - self.means[k]
        #    self.covariances[k] = np.dot((responsibilities[:, k] * X_centered).T, X_centered) / np.sum(responsibilities[:, k])

        for k in range(self.n_components):
            X_centered = X - self.means[k]
            # 修改这一行：确保responsibilities[:, k]是一个列向量
            resp_reshaped = responsibilities[:, k][:, np.newaxis]
            self.covariances[k] = np.dot((resp_reshaped * X_centered).T, X_centered) / np.sum(responsibilities[:, k])

    
    def _log_likelihood(self, X):
        likelihood = np.sum([self.weights[k] * multivariate_normal.pdf(X, self.means[k], self.covariances[k])
                             for k in range(self.n_components)], axis=0)
        return np.sum(np.log(likelihood))

    def predict(self, X):
        return np.argmax(self._e_step(X), axis=1)

if __name__ == '__main__':
    # 创建数据
    X = np.random.randn(300, 2)

    # 初始化模型
    gmm = GaussianMixtureModel(n_components=2, n_iter=100, tol=1e-4)

    # 拟合数据
    gmm.fit(X)

    # 预测数据点的类别
    labels = gmm.predict(X)

    # 可视化原始数据
    plt.figure(figsize=(10, 5))

    # 左侧图显示原始数据
    plt.subplot(1, 2, 1)
    plt.scatter(X[:, 0], X[:, 1], color='gray', label='Data Points')
    plt.title('Original Data')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()

    # 右侧图显示聚类结果
    plt.subplot(1, 2, 2)
    for i, color in enumerate(['blue', 'green', 'red', 'purple', 'orange', 'yellow']):  # 更多颜色以支持多个聚类
        if i >= gmm.n_components:
            break
        plt.scatter(X[labels == i, 0], X[labels == i, 1], color=color, label=f'Cluster {i+1}')
    plt.scatter(gmm.means[:, 0], gmm.means[:, 1], color='black', marker='x', s=100, label='Centroids')  # 中心点
    plt.title('Clustered Data (GMM)')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()

    plt.tight_layout()
    plt.show()

