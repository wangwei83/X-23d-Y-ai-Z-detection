import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.ndimage import convolve

# Let's assume the uploaded image is named 'input_image.jpg' in the working directory.
# I will use a placeholder image since I cannot access files directly.
# You would need to upload the actual image and use the correct path here.
input_image_path = '.\lesson5\mouse.png'  # This should be the path to the actual image file.

# Load the input image
input_image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)  # Convert image to grayscale

# Define the convolution kernel
kernel = np.array([
    [-1, -1, -1],
    [-1,  8, -1],
    [-1, -1, -1]
])

# Apply convolution using the defined kernel
feature_map = convolve(input_image.astype(float), kernel, mode='constant', cval=0.0)

# Plotting the input image, the kernel and the feature map
fig, axes = plt.subplots(1, 3, figsize=(10, 5))

# Input image
axes[0].imshow(input_image, cmap='gray')
axes[0].set_title('Input image')
axes[0].axis('off')  # Hide axis

# Convolution Kernel as an image
# I am displaying the kernel as an image with text, assuming it's small enough.
kernel_img = np.full((50, 50), 255)  # White background
axes[1].imshow(kernel_img, cmap='gray')
axes[1].axis('off')  # Hide axis
for (j, i), label in np.ndenumerate(kernel):
    axes[1].text(i*16+10, j*16+20, str(label), ha='center', va='center', color='black')
axes[1].set_title('Convolution Kernel')

# Feature map
axes[2].imshow(feature_map, cmap='gray')
axes[2].set_title('Feature map')
axes[2].axis('off')  # Hide axis

plt.tight_layout()
plt.show()
