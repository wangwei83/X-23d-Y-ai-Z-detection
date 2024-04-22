import numpy as np
from scipy.spatial import KDTree
import matplotlib.pyplot as plt

# 生成一些随机数据点
np.random.seed(0)
points = np.random.random((10, 2))  # 10个二维空间的点

# 构建KD树
tree = KDTree(points)

# 可视化点和KD树的分割
fig, ax = plt.subplots()
ax.plot(points[:, 0], points[:, 1], 'bo')  # 画点

def plot_node(node, min_x, max_x, min_y, max_y, depth=0):
    if node is None:
        return

    # Check if the current node is a leaf node
    if not hasattr(node, 'split'):
        return  # Leaf node, no split attribute, just return

    # Calculate vertical or horizontal division
    axis = depth % 2
    if axis == 0:
        ax.plot([node.split, node.split], [min_y, max_y], '-r')
        plot_node(node.less, min_x, node.split, min_y, max_y, depth + 1)
        plot_node(node.greater, node.split, max_x, min_y, max_y, depth + 1)
    else:
        ax.plot([min_x, max_x], [node.split, node.split], '-r')
        plot_node(node.less, min_x, max_x, min_y, node.split, depth + 1)
        plot_node(node.greater, min_x, max_x, node.split, max_y, depth + 1)


plot_node(tree.tree, 0, 1, 0, 1)

ax.set_title('KD Tree')
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
plt.show()
