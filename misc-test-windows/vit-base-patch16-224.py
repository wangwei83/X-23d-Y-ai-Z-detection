'''
Author: wangwei83 wangwei83@cuit.edu.cn
Date: 2024-06-05 23:14:20
LastEditors: wangwei83 wangwei83@cuit.edu.cn
LastEditTime: 2024-06-05 23:20:50
FilePath: /wangwei/X-23d-Y-ai-Z-detection/misc-test-windows/vit-base-patch16-224.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
# https://huggingface.co/google/vit-base-patch16-224
# https://huggingface.co/google/vit-base-patch16-224/tree/main/pytorch_model.bin


from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import requests
url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)

processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')