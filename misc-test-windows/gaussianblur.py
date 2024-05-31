import cv2
import numpy as np

# 读取图像
image_path = '3D.jpg'
image = cv2.imread(image_path)

if image is None:
    raise FileNotFoundError(f"The image file {image_path} was not found.")

# 对图像进行高斯模糊
blurred = cv2.GaussianBlur(image, (5, 5), 0)

# 显示原图像和模糊后的图像
cv2.imshow('Original Image', image)
cv2.imshow('Blurred Image', blurred)

cv2.waitKey(0)
cv2.destroyAllWindows()