import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np

class KDNode:
    def __init__(self, point, depth=0, left=None, right=None):
        self.point = point
        self.depth = depth
        self.left = left
        self.right = right

def build_kdtree(points, depth=0):
    # Check if the list of points is empty using a more appropriate method
    if len(points) == 0:
        return None

    k = len(points[0])  # Assumes all points have the same dimension
    axis = depth % k   # Alternate between axes based on depth

    # Sort point list by the current axis and choose the median
    points = sorted(points, key=lambda x: x[axis])
    median = len(points) // 2

    # Recursively build the tree
    return KDNode(
        point=points[median],
        depth=depth,
        left=build_kdtree(points[:median], depth + 1),
        right=build_kdtree(points[median + 1:], depth + 1)
    )

# Example usage:
# Generate some random 3D points
points = np.random.rand(10, 3) * 10
tree = build_kdtree(points)

# Assuming there's more code to visualize or interact with the tree


def visualize_kdtree(node, ax, min_point, max_point):
    if node is not None:
        # Determine the axis and the splitting coordinate
        axis = node.depth % len(node.point)
        point = node.point

        if axis == 0:  # Splitting x-axis
            ax.plot([point[0], point[0]], [min_point[1], max_point[1]], [min_point[2], max_point[2]], 'r')
        elif axis == 1:  # Splitting y-axis
            ax.plot([min_point[0], max_point[0]], [point[1], point[1]], [min_point[2], max_point[2]], 'g')
        else:  # Splitting z-axis
            ax.plot([min_point[0], max_point[0]], [min_point[1], max_point[1]], [point[2], point[2]], 'b')

        # Next branches
        next_min = min_point.copy()
        next_max = max_point.copy()
        next_min[axis] = point[axis]
        next_max[axis] = point[axis]

        visualize_kdtree(node.left, ax, min_point, next_max)
        visualize_kdtree(node.right, ax, next_min, max_point)

        ax.scatter(*point, c='k')

points = np.random.rand(10, 3) * 10  # Generate some random points in 3D space
tree = build_kdtree(points)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_title('3D KD-Tree Visualization')
visualize_kdtree(tree, ax, np.min(points, axis=0), np.max(points, axis=0))
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()
