import cv2
from matplotlib import pyplot as plt

# 读取图像，0表示以灰度模式读取  3D视觉缺陷检测\lesson2\histogram.jpg
image = cv2.imread('.\3D视觉缺陷检测\lesson2\histogram.jpghistogram.jpg', 0)

# 应用直方图均衡化
equ = cv2.equalizeHist(image)


