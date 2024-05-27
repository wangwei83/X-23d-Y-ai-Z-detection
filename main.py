import cv2
import numpy as np

# 加载图像
img = cv2.imread('X-23d-Y-ai-Z-detection/image/人工智能点云处理机深度学习算法.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Canny 边缘检测
edges = cv2.Canny(gray, 50, 150, apertureSize=3)

# 霍夫线变换
lines = cv2.HoughLines(edges, 1, np.pi/180, 200)

# 在图像上绘制线
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

    cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

# 保存图像
cv2.imwrite('X-23d-Y-ai-Z-detection/image/output.jpg', img)