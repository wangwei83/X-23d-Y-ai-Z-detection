import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# 定义模型函数
def model(x, a, b):
    return a * np.exp(b * x)

# 生成一些模拟数据
x_data = np.linspace(0, 4, 50)
a_actual = 2.5
b_actual = -1.3
y_data = model(x_data, a_actual, b_actual) + 0.5 * np.random.normal(size=x_data.size)

# 使用curve_fit函数来拟合模型
popt, pcov = curve_fit(model, x_data, y_data)

# 使用拟合得到的参数生成拟合曲线
y_fit = model(x_data, *popt)

# 绘制数据点
plt.scatter(x_data, y_data, label='Data (noisy)', color='red', marker='o')

# 绘制拟合曲线
plt.plot(x_data, y_fit, label='Fit: a=%5.3f, b=%5.3f' % tuple(popt), color='blue')

# 可选：如果知道真实参数，也可以绘制真实模型曲线
y_true = model(x_data, a_actual, b_actual)
plt.plot(x_data, y_true, label='True model', color='green', linestyle='--')

# 添加图例
plt.legend()

# 添加标题和坐标轴标签
plt.title('Non-linear Least Squares Fit')
plt.xlabel('x')
plt.ylabel('y')

# 显示图形
plt.show()
