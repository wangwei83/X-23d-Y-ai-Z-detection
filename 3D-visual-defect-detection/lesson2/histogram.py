import cv2
from matplotlib import pyplot as plt

# 读取图像，0表示以灰度模式读取  C:\Users\19002\Desktop\cloud_lesson\3D-visual-defect-detection\lesson2\histogram.jpg
image = cv2.imread(r'C:\Users\19002\Desktop\cloud_lesson\3D-visual-defect-detection\lesson2\histogram.jpg', 0)

# 应用直方图均衡化
equ = cv2.equalizeHist(image)


# 应用直方图均衡化
equ = cv2.equalizeHist(image)

# 将原始图像和处理后的图像并排显示
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')

plt.subplot(1, 2, 2)
plt.imshow(equ, cmap='gray')
plt.title('Equalized Image')

plt.show()