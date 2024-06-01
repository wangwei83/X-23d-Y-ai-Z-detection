'''
Author: wangwei83 wangwei83@cuit.edu.cn
Date: 2024-05-31 23:45:25
LastEditors: wangwei83 wangwei83@cuit.edu.cn
LastEditTime: 2024-06-01 08:28:23
FilePath: /wangwei/X-23d-Y-ai-Z-detection/misc-test-windows/timm-install-test.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
from transformers import AutoImageProcessor, ResNetForImageClassification
import torch
from datasets import load_dataset
from PIL import Image
import numpy as np

# dataset = load_dataset("huggingface/cats-image")
# image = dataset["test"]["image"][0]

# 读取图像
image = Image.open("cats_image.jpeg")
# 将 PIL 图像转换为 NumPy 数组
image = np.array(image)


processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")

inputs = processor(image, return_tensors="pt")

with torch.no_grad():
    logits = model(**inputs).logits

# model predicts one of the 1000 ImageNet classes
predicted_label = logits.argmax(-1).item()
print(model.config.id2label[predicted_label])
