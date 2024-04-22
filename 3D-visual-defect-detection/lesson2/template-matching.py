import cv2
import numpy as np
from matplotlib import pyplot as plt

# 读取大图和模板图
main_image = cv2.imread(r'3D-visual-defect-detection\lesson2\histogram.jpg', 0)  # 灰度模式
template = cv2.imread(r'C:\Users\19002\Desktop\cloud_lesson\3D-visual-defect-detection\lesson2\template.jpg', 0)  # 灰度模式
h, w = template.shape[:2]  # 获取模板图的高度和宽度

# 进行模板匹配
res = cv2.matchTemplate(main_image, template, cv2.TM_CCOEFF_NORMED)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

# TM_CCOEFF_NORMED方法返回最大相关性的位置
top_left = max_loc
bottom_right = (top_left[0] + w, top_left[1] + h)

# 在大图中画出模板的位置
cv2.rectangle(main_image, top_left, bottom_right, 255, 2)

# 显示结果
plt.imshow(main_image, cmap='gray')
plt.title('Detected Area')
plt.show()
