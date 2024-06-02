'''
Author: wangwei83 wangwei83@cuit.edu.cn
Date: 2024-06-02 22:06:32
LastEditors: wangwei83 wangwei83@cuit.edu.cn
LastEditTime: 2024-06-02 22:06:45
FilePath: /wangwei/X-23d-Y-ai-Z-detection/misc-test-windows/timm_resnet.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
from urllib.request import urlopen
from PIL import Image
import timm
import torch

img = Image.open(urlopen(
    'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png'
))

model = timm.create_model('resnetv2_152x2_bit.goog_teacher_in21k_ft_in1k', pretrained=True)
model = model.eval()

# get model specific transforms (normalization, resize)
data_config = timm.data.resolve_model_data_config(model)
transforms = timm.data.create_transform(**data_config, is_training=False)

output = model(transforms(img).unsqueeze(0))  # unsqueeze single image into batch of 1

top5_probabilities, top5_class_indices = torch.topk(output.softmax(dim=1) * 100, k=5)
