import torch
import matplotlib.pyplot as plt
from torch_geometric.datasets import KarateClub
from torch_geometric.nn import GCNConv
from sklearn.manifold import TSNE
import networkx as nx
from torch_geometric.utils import to_networkx

# 加载数据集
dataset = KarateClub()

# 提取数据和标签
data = dataset[0]
labels = data.y

# 定义图神经网络
class GCN(torch.nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(dataset.num_node_features, 4)
        self.conv2 = GCNConv(4, dataset.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return x, torch.softmax(x, dim=1)

# 初始化网络和优化器
model = GCN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 简单的训练循环
def train(data):
    optimizer.zero_grad()
    out, _ = model(data)
    loss = torch.nn.functional.nll_loss(out, labels)
    loss.backward()
    optimizer.step()
    return loss

for epoch in range(200):
    loss = train(data)
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

# 获取训练后的节点嵌入
_, embeddings = model(data)

# 转换为2D空间
tsne = TSNE(n_components=2)
embeddings_2d = tsne.fit_transform(embeddings.detach().numpy())

# 可视化
def visualize(embeddings, color):
    plt.figure(figsize=(7, 7))
    plt.scatter(embeddings[:, 0], embeddings[:, 1], c=color, cmap="viridis")
    plt.show()

visualize(embeddings_2d, color=data.y.numpy())

# 可视化图结构
G = to_networkx(data, to_undirected=True)
nx.draw(G, with_labels=True, node_color=data.y.numpy(), cmap="viridis")
plt.show()

