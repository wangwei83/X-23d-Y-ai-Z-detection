import numpy as np
import matplotlib.pyplot as plt

class KDNode:
    def __init__(self, point, depth=0, left=None, right=None):
        self.point = point
        self.depth = depth
        self.left = left
        self.right = right

def build_kdtree(points, depth=0):
    if not points:
        return None

    k = len(points[0])  # Assumes all points have the same dimension
    axis = depth % k   # Alternate between the x and y axis

    points.sort(key=lambda x: x[axis])
    median = len(points) // 2

    return KDNode(
        point=points[median],
        depth=depth,
        left=build_kdtree(points[:median], depth + 1),
        right=build_kdtree(points[median + 1:], depth + 1)
    )

def visualize_kdtree(node, ax, xmin, xmax, ymin, ymax):
    if node is not None:
        x, y = node.point
        if node.depth % 2 == 0:  # Vertical line
            ax.plot([x, x], [ymin, ymax], 'k-')
            visualize_kdtree(node.left, ax, xmin, x, ymin, ymax)
            visualize_kdtree(node.right, ax, x, xmax, ymin, ymax)
        else:  # Horizontal line
            ax.plot([xmin, xmax], [y, y], 'k-')
            visualize_kdtree(node.left, ax, xmin, xmax, ymin, y)
            visualize_kdtree(node.right, ax, xmin, xmax, y, ymax)
        
        ax.plot(x, y, 'ko')

points = [(7, 2), (5, 4), (9, 6), (2, 3), (4, 7), (8, 1)]
tree = build_kdtree(points)

fig, ax = plt.subplots()
ax.set_title('2D KD-Tree Visualization')
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
visualize_kdtree(tree, ax, 0, 10, 0, 10)
plt.show()
