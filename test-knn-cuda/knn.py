import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# 生成示例数据
# X 是特征，y 是标签
X = np.array([[1, 2], [2, 3], [3, 4], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12]])
y = np.array([0, 0, 0, 1, 1, 1, 1, 1, 0, 0])

# 将数据集分成训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 创建KNN分类器，k设为3
knn = KNeighborsClassifier(n_neighbors=3)

# 使用训练集训练模型
knn.fit(X_train, y_train)

# 使用测试集进行预测
y_pred = knn.predict(X_test)

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# 输出测试集的预测结果
print(f'Test set predictions: {y_pred}')
