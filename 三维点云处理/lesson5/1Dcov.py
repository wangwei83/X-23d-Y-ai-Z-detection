import numpy as np

# Define the 1D convolution function
def conv1d(x, w, stride=1):
    # Compute the length of the output after convolution
    output_length = ((len(x) - len(w)) // stride) + 1
    # Initialize the output with zeros
    y = np.zeros(output_length)
    # Perform the convolution operation
    for i in range(output_length):
        y[i] = np.sum(x[i*stride:i*stride+len(w)] * w)
    return y


# Updated inputs
x_updated = np.array([1, 0, 2, 2, 3, 1, 0])  # Updated input array (x^T)
w = np.array([3, 2, 2])                      # Weights (w^T) remain the same

# Perform 1D convolution with the updated input
y_updated = conv1d(x_updated, w)

# Output the result of the convolution
print(y_updated)
