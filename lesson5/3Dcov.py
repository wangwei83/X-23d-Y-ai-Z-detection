import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve

# 生成一个简单的3D数据块
data = np.random.rand(4, 4, 4)

# 定义一个3D卷积核
kernel = np.array([[[1, -1], [-1, 1]], 
                   [[-1, 1], [1, -1]]])

# 应用3D卷积
convolved_data = convolve(data, kernel)

# 可视化原始数据块的第一层
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.imshow(data[:, :, 0], cmap='viridis')
plt.title('Original Data Slice')
plt.axis('off')

# 可视化卷积后数据块的第一层
plt.subplot(1, 2, 2)
plt.imshow(convolved_data[:, :, 0], cmap='viridis')
plt.title('Convolved Data Slice')
plt.axis('off')

plt.show()
