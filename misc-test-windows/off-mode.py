'''
Author: wangwei83 wangwei83@cuit.edu.cn
Date: 2024-06-01 22:37:15
LastEditors: wangwei83 wangwei83@cuit.edu.cn
LastEditTime: 2024-06-01 22:40:32
FilePath: /wangwei/X-23d-Y-ai-Z-detection/misc-test-windows/off-mode.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''


from transformers import T5Model

model = T5Model.from_pretrained("./path/to/local/directory", local_files_only=True)
input_text = "Hello, world!"
output = model.generate(input_text)
print(output)