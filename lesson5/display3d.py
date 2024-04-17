import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Let's create a meshgrid of values
x = np.linspace(-1.0, 1.0, 100)
y = np.linspace(-1.0, 1.0, 100)
X, Y = np.meshgrid(x, y)
Z = np.sinc(np.sqrt(X**2 + Y**2))  # Using the sinc function as an example

# Create a 3D plot
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Plot the surface
surf = ax.plot_surface(X, Y, Z, cmap='coolwarm', edgecolor='none')

# Dummy points for legend - one for each optimizer
# These are not the real paths, just placeholders for the legend
optimizers = ['SGD', 'Momentum', 'NAG', 'Adagrad', 'Adadelta', 'Rmsprop']
for optimizer in optimizers:
    ax.scatter([], [], [], label=optimizer)

# Customize the z axis.
ax.set_zlim(-2.01, 1.01)
ax.zaxis.set_major_locator(plt.LinearLocator(10))
ax.zaxis.set_major_formatter(plt.FormatStrFormatter('%.02f'))

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

# Legend
plt.legend()

# Show the plot
plt.show()
