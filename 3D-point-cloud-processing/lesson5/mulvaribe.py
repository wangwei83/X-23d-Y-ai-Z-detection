import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 设置随机种子以获得可重复结果
np.random.seed(42)

# 生成模拟数据
num_samples = 500
square_feet = np.random.normal(1500, 250, num_samples).astype(int)  # 均值为1500，标准差为250
bedrooms = np.random.randint(1, 5, num_samples)  # 1到4卧室
bathrooms = np.random.randint(1, 3, num_samples)  # 1到2浴室
price = square_feet * 150 + bedrooms * 40000 + bathrooms * 30000 + np.random.normal(0, 25000, num_samples)

# 创建DataFrame
data = pd.DataFrame({
    'SquareFeet': square_feet,
    'Bedrooms': bedrooms,
    'Bathrooms': bathrooms,
    'Price': price
})

# 保存到CSV文件
data.to_csv('.\lesson5\house_data.csv', index=False)

# 读取数据
data = pd.read_csv('.\lesson5\house_data.csv')

# 定义特征和目标变量
X = data[['SquareFeet', 'Bedrooms', 'Bathrooms']]
y = data['Price']

# 分割数据为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 创建线性回归模型
model = LinearRegression()

# 训练模型
model.fit(X_train, y_train)

# 预测测试集的房价
y_pred = model.predict(X_test)

# 计算和打印模型性能指标
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")

# 可视化实际价格和预测价格
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.75, color='b', label='Predicted vs Actual')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual Prices vs Predicted Prices')
plt.legend()
plt.show()

# 可视化误差分布
plt.figure(figsize=(10, 6))
errors = y_pred - y_test
plt.hist(errors, bins=25, color='purple', alpha=0.7)
plt.xlabel('Prediction Error')
plt.ylabel('Frequency')
plt.title('Prediction Error Distribution')
plt.show()
