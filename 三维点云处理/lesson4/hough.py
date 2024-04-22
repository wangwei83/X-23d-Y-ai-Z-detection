import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图像
image = cv2.imread('C:/Users/19002/Desktop/cloud_lesson/lesson4/robotic-guidance-is-3800.jpg')  #C:\Users\19002\Desktop\cloud_lesson\lesson4\robotic-guidance-is-3800.jpg
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 边缘检测
edges = cv2.Canny(gray, 50, 150, apertureSize=3)

# 使用霍夫变换进行直线检测
lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)

# 绘制直线
for line in lines:
    rho, theta = line[0]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * (a))
    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * (a))

    cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

# 显示图像
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Detected Lines')
plt.show()
