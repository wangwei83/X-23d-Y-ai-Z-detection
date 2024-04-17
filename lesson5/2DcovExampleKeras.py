import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from skimage import data
import numpy as np

# Load an example image
rgb_image = data.astronaut()

# Display the original image
plt.imshow(rgb_image)
plt.title('Original Image')
plt.axis('off')  # Hide the axes
plt.show()

# Create a model with a single convolutional layer with one filter
model = Sequential([
    Conv2D(filters=1, kernel_size=(3, 3), strides=(1, 1), padding='valid', input_shape=(rgb_image.shape[0], rgb_image.shape[1], 3))
])

# Initialize the weights of the filter with random numbers
weights = model.layers[0].get_weights()
weights[0][:, :, :, 0] = np.random.rand(3, 3, 3) * 2 - 1  # random weights between -1 and 1
model.layers[0].set_weights(weights)

# Reshape image for the model (add batch dimension)
image_batch = np.expand_dims(rgb_image, axis=0)

# Apply convolution to the image
conv_image = model.predict(image_batch)

# Squeeze the batch dimension and the channels dimension since we have a single filter
conv_image = np.squeeze(conv_image, axis=0)
conv_image = np.squeeze(conv_image, axis=2)

# Display the convolved image
plt.imshow(conv_image, cmap='gray')
plt.title('Convolved Image')
plt.axis('off')  # Hide the axes
plt.show()
