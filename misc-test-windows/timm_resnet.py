'''
Author: wangwei83 wangwei83@cuit.edu.cn
Date: 2024-06-02 22:06:32
LastEditors: wangwei83 wangwei83@cuit.edu.cn
LastEditTime: 2024-06-03 10:30:42
FilePath: /wangwei/X-23d-Y-ai-Z-detection/misc-test-windows/timm_resnet.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
# https://huggingface.co/timm/resnetv2_152x2_bit.goog_teacher_in21k_ft_in1k_384

import timm
from timm.models.efficientnet import _cfg
from PIL import Image
import torch

img=Image.open('/data/wangwei/X-23d-Y-ai-Z-detection/misc-test-windows/beignets-task-guide.png')

config = _cfg(url='', file='/data/wangwei/resnetv2_152x2_bit.bin') #file为本地文件路径

model = timm.create_model(
                                "resnetv2_152x2_bit.goog_teacher_in21k_ft_in1k",
                                pretrained=True,
                                features_only=True,
                                pretrained_cfg=config
                            )
model = model.eval()

# get model specific transforms (normalization, resize)
data_config = timm.data.resolve_model_data_config(model)
transforms = timm.data.create_transform(**data_config, is_training=False)

output = model(transforms(img).unsqueeze(0))  # unsqueeze single image into batch of 1

output_tensor = output[0]  # select the first feature
output_tensor = torch.tensor(output_tensor)  # convert the feature to a tensor
top5_probabilities, top5_class_indices = torch.topk(output_tensor.softmax(dim=1) * 100, k=5)

for o in output:
    # print shape of each feature map in output
    # e.g.:
    #  torch.Size([1, 128, 192, 192])
    #  torch.Size([1, 512, 96, 96])
    #  torch.Size([1, 1024, 48, 48])
    #  torch.Size([1, 2048, 24, 24])
    #  torch.Size([1, 4096, 12, 12])

    print(o.shape)

# or equivalently (without needing to set num_classes=0)

output = model.forward(transforms(img).unsqueeze(0))
