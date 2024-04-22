import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import RANSACRegressor
from sklearn.datasets import make_regression

# 生成回归数据
X, y, coef = make_regression(n_samples=200, n_features=1, noise=10.0,
                             coef=True, random_state=0)

# 添加异常值
np.random.seed(42)
n_outliers = 50
X[:n_outliers] = 3 + 0.5 * np.random.normal(size=(n_outliers, 1))
y[:n_outliers] = -3 + 10 * np.random.normal(size=n_outliers)

# 应用RANSAC算法
ransac = RANSACRegressor(min_samples=50, residual_threshold=5, max_trials=100)
ransac.fit(X, y)
inlier_mask = ransac.inlier_mask_
outlier_mask = np.logical_not(inlier_mask)

# 预测数据
line_X = np.arange(X.min(), X.max())[:, np.newaxis]
line_y_ransac = ransac.predict(line_X)

# 绘制结果
plt.scatter(X[inlier_mask], y[inlier_mask], color='yellowgreen', marker='.',
            label='Inliers')
plt.scatter(X[outlier_mask], y[outlier_mask], color='gold', marker='.',
            label='Outliers')
plt.plot(line_X, line_y_ransac, color='cornflowerblue', linewidth=2,
         label='RANSAC regressor')
plt.legend()
plt.xlabel('Input')
plt.ylabel('Response')
plt.title('RANSAC Regression')
plt.show()
