<!--
 * @Author: wangwei83 wangwei83@cuit.edu.cn
 * @Date: 2024-06-03 18:49:15
 * @LastEditors: wangwei83 wangwei83@cuit.edu.cn
 * @LastEditTime: 2024-06-03 18:49:35
 * @FilePath: /wangwei/X-23d-Y-ai-Z-detection/misc-test-windows/misc-test-windows.md
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
-->


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

data_config = timm.data.resolve_model_data_config(model)
print(data_config)
transforms = timm.data.create_transform(**data_config, is_training=False)

output = model(transforms(img).unsqueeze(0))  # unsqueeze single image into batch of 1

output_tensor = output[0]  # select the first feature
output_tensor = torch.tensor(output_tensor)  # convert the feature to a tensor
top5_probabilities, top5_class_indices = torch.topk(output_tensor.softmax(dim=1) * 100, k=5)
print(top5_probabilities)
print(top5_class_indices)
print(output)