import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def load_image(image_path):
    """加载图像并转换为灰度图像"""
    with Image.open(image_path) as img:
        img_gray = img.convert('L')  # 转换为灰度图
    return np.array(img_gray)

def compress_image(image, k):
    """使用前k个奇异值来压缩图像"""
    U, s, Vt = np.linalg.svd(image, full_matrices=False)
    S = np.diag(s[:k])
    U_k = U[:, :k]
    Vt_k = Vt[:k, :]
    compressed_image = np.dot(U_k, np.dot(S, Vt_k))
    return compressed_image

def plot_images(original, compressed):
    """绘制原始图像和压缩后的图像"""
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(original, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    axes[1].imshow(compressed, cmap='gray')
    axes[1].set_title('Compressed Image')
    axes[1].axis('off')

    plt.show()

# 加载图像
image_path = r'C:\Users\19002\Desktop\cloud_lesson\3D-point-cloud-processing\lesson9\企业申报政府基金.jpg'  # 修改为你的图片路径
image = load_image(image_path)

# 压缩图像
k = 50  # 保留的奇异值数量
compressed_image = compress_image(image, k)

# 显示图像
plot_images(image, compressed_image)
