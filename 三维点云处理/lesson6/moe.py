import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

class MoELayer(nn.Module):
    def __init__(self, num_experts, in_features, out_features):
        super(MoELayer, self).__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList([nn.Linear(in_features, out_features) for _ in range(num_experts)])
        self.gate = nn.Linear(in_features, num_experts)

    def forward(self, x):
        # 获取每个专家的权重
        gates = self.gate(x)
        weights = torch.softmax(gates, dim=1)

        # 获得专家的输出
        expert_outputs = [expert(x) for expert in self.experts]
        expert_outputs = torch.stack(expert_outputs, dim=1)

        # 加权求和
        output = torch.einsum('be,bei->bi', weights, expert_outputs)
        return output

class SimpleMoENet(nn.Module):
    def __init__(self, num_experts, in_features, hidden_features, out_features):
        super(SimpleMoENet, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.moe = MoELayer(num_experts, hidden_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.moe(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x

# 创建模型和数据示例
num_classes = 10
model = SimpleMoENet(num_experts=3, in_features=784, hidden_features=256, out_features=num_classes)
input = torch.randn(1, 784)
output = model(input)

print("Model output:", output)
