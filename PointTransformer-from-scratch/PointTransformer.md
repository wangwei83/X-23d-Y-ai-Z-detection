<!--
 * @Author: wangwei83 wangwei83@cuit.edu.cn
 * @Date: 2024-06-16 21:39:57
 * @LastEditors: wangwei83 wangwei83@cuit.edu.cn
 * @LastEditTime: 2024-06-16 22:10:50
 * @FilePath: /wangwei/X-23d-Y-ai-Z-detection/PointTransformer-from-scratch/PointTransformer.md
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
-->


在这项工作中，我们提出了Point Transformer，这是一种直接在无序和非结构化点集上运行的深度神经网络。我们设计了Point Transformer来提取局部和全局特征，并通过引入局部-全局注意机制来关联这两种表示，该机制旨在捕捉空间点关系和形状信息。为此，我们提出了SortNet，作为Point Transformer的一部分，通过选择基于学习得分的点来引入输入排列不变性。Point Transformer的输出是一个排序且排列不变的特征列表，可以直接整合到常见的计算机视觉应用中。我们在标准分类和部件分割基准上评估了我们的方法，展示了相对于先前工作的竞争结果。

这段文本描述了一种名为 Point Transformer 的深度学习模型，它专为处理无序和非结构化的点集而设计。Point Transformer 的核心特点包括：

1. **局部和全局特征提取**：设计用于同时提取点集中的局部细节和全局结构信息。
2. **局部-全局注意机制**：通过这种机制，模型能够关联局部和全局表示，以捕捉点之间的空间关系和形状信息。
3. **SortNet**：为了保证模型对输入点的排列不变性，引入了 SortNet，它基于学习得分选择点，确保 Point Transformer 的输出特征列表是排序且排列不变的。

Point Transformer 的设计使其能够直接应用于各种计算机视觉任务，如分类和部件分割，并在这些任务的标准基准测试中展示了与先前工作相比的竞争性能。


wget https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip -P data/ --no-check-certificate

# 下载 ModelNet40 数据集并解压到 data 文件夹
wget https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip -P data/
unzip data/modelnet40_normal_resampled.zip -d data/

# 下载 ShapeNet 数据集并解压到 data 文件夹
wget https://shapenet.cs.stanford.edu/media/shapenetcore_partanno_segmentation_benchmark_v0_normal.zip -P data/
unzip data/shapenetcore_partanno_segmentation_benchmark_v0_normal.zip -d data/

# 安装 Python 依赖
pip install -r requirements.txt