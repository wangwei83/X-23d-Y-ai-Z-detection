# cloud_lesson

conda create -n cloud_lesson

conda activate cloud_lesson

conda install python=3.11

pip install open3d

pip install pyntcloud

如果没有打印和现实，明天在崇州那台电脑上进一步调试
可以显示了。但是不知道为什么不能打印print，准确说是在o3d调用之后。

[Open3D WARNING] [ViewControl] SetViewPoint() failed because window height and width are not set.

pip install jupyterlab

jupyter lab


--------------------------
最临近问题: 无处不在，法向量估计，噪音去除，上采样下采样，聚类，深度学习，特征提取
binary search tree
kd-tree
octree

k-nn
fixed radius-NN

点云上的最近邻问题，比图像更难，
稀疏，
三维，网格很多，组成的网格的内存就大，网格大部分区域是空的。
点云的数据量通常比较大。
暴力搜索60亿次计算，20hz
05年的超算来处理velodyne的点云都是比较困难

停止条件，选择树
分割空间，跳跃空间，停止搜索

迭代、递归实现BST的查找

