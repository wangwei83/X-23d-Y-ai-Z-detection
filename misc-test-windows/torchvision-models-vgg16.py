'''
Author: wangwei83 wangwei83@cuit.edu.cn
Date: 2024-05-31 21:42:00
LastEditors: wangwei83 wangwei83@cuit.edu.cn
LastEditTime: 2024-05-31 21:53:13
FilePath: /wangwei/X-23d-Y-ai-Z-detection/misc-test-windows/timm-test.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import torch
from torchvision import models, transforms
from PIL import Image

# 加载预训练的 VGG16 模型
model = models.vgg16(pretrained=True)
model.eval()

# 图片预处理
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 加载图片
image = Image.open("OIP-1.jpg")  # misc-test-windows\OIP.jpg
input_tensor = preprocess(image) 
input_batch = input_tensor.unsqueeze(0)

# 如果有GPU，将模型和数据都移动到GPU上
if torch.cuda.is_available():
    model = model.to('cuda')
    input_batch = input_batch.to('cuda')

# 预测
with torch.no_grad():
    output = model(input_batch)

# 获取预测结果
_, predicted_idx = torch.max(output, 1)
print("Predicted:", predicted_idx.item())
# print("Predicted class:", model.classifier[predicted_idx.item()])