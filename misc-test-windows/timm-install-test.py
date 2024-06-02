'''
Author: wangwei83 wangwei83@cuit.edu.cn
Date: 2024-05-31 23:45:25
LastEditors: wangwei83 wangwei83@cuit.edu.cn
LastEditTime: 2024-06-01 23:05:54
FilePath: /wangwei/X-23d-Y-ai-Z-detection/misc-test-windows/timm-install-test.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
# from transformers import AutoImageProcessor, ResNetForImageClassification
# import torch
# from datasets import load_dataset
# from PIL import Image
# import numpy as np

# # dataset = load_dataset("huggingface/cats-image")
# # image = dataset["test"]["image"][0]

# # 读取图像
# image = Image.open("cats_image.jpeg")
# # 将 PIL 图像转换为 NumPy 数组
# image = np.array(image)


# processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
# model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")

# inputs = processor(image, return_tensors="pt")

# with torch.no_grad():
#     logits = model(**inputs).logits

# # model predicts one of the 1000 ImageNet classes
# predicted_label = logits.argmax(-1).item()
# print(model.config.id2label[predicted_label])

# (M3DM) (base) c206@c206linux:/data/wangwei/X-23d-Y-ai-Z-detection/misc-test-windows$ python timm-install-test.py 
# Traceback (most recent call last):
#   File "/home/c206/anaconda3/envs/M3DM/lib/python3.8/site-packages/urllib3/connection.py", line 198, in _new_conn
#     sock = connection.create_connection(
#   File "/home/c206/anaconda3/envs/M3DM/lib/python3.8/site-packages/urllib3/util/connection.py", line 85, in create_connection
#     raise err
#   File "/home/c206/anaconda3/envs/M3DM/lib/python3.8/site-packages/urllib3/util/connection.py", line 73, in create_connection
#     sock.connect(sa)
# OSError: [Errno 101] Network is unreachable

# The above exception was the direct cause of the following exception:

# Traceback (most recent call last):
#   File "/home/c206/anaconda3/envs/M3DM/lib/python3.8/site-packages/urllib3/connectionpool.py", line 793, in urlopen
#     response = self._make_request(
#   File "/home/c206/anaconda3/envs/M3DM/lib/python3.8/site-packages/urllib3/connectionpool.py", line 491, in _make_request
#     raise new_e
#   File "/home/c206/anaconda3/envs/M3DM/lib/python3.8/site-packages/urllib3/connectionpool.py", line 467, in _make_request
#     self._validate_conn(conn)
#   File "/home/c206/anaconda3/envs/M3DM/lib/python3.8/site-packages/urllib3/connectionpool.py", line 1099, in _validate_conn
#     conn.connect()
#   File "/home/c206/anaconda3/envs/M3DM/lib/python3.8/site-packages/urllib3/connection.py", line 616, in connect
#     self.sock = sock = self._new_conn()
#   File "/home/c206/anaconda3/envs/M3DM/lib/python3.8/site-packages/urllib3/connection.py", line 213, in _new_conn
#     raise NewConnectionError(
# urllib3.exceptions.NewConnectionError: <urllib3.connection.HTTPSConnection object at 0x7fb8c5bf4910>: Failed to establish a new connection: [Errno 101] Network is unreachable

# The above exception was the direct cause of the following exception:

# Traceback (most recent call last):
#   File "/home/c206/anaconda3/envs/M3DM/lib/python3.8/site-packages/requests/adapters.py", line 589, in send
#     resp = conn.urlopen(
#   File "/home/c206/anaconda3/envs/M3DM/lib/python3.8/site-packages/urllib3/connectionpool.py", line 847, in urlopen
#     retries = retries.increment(
#   File "/home/c206/anaconda3/envs/M3DM/lib/python3.8/site-packages/urllib3/util/retry.py", line 515, in increment
#     raise MaxRetryError(_pool, url, reason) from reason  # type: ignore[arg-type]
# urllib3.exceptions.MaxRetryError: HTTPSConnectionPool(host='huggingface.co', port=443): Max retries exceeded with url: /microsoft/resnet-50/resolve/main/preprocessor_config.json (Caused by NewConnectionError('<urllib3.connection.HTTPSConnection object at 0x7fb8c5bf4910>: Failed to establish a new connection: [Errno 101] Network is unreachable'))

# During handling of the above exception, another exception occurred:

# Traceback (most recent call last):
#   File "/home/c206/anaconda3/envs/M3DM/lib/python3.8/site-packages/huggingface_hub/file_download.py", line 1722, in _get_metadata_or_catch_error
#     metadata = get_hf_file_metadata(url=url, proxies=proxies, timeout=etag_timeout, headers=headers)
#   File "/home/c206/anaconda3/envs/M3DM/lib/python3.8/site-packages/huggingface_hub/utils/_validators.py", line 114, in _inner_fn
#     return fn(*args, **kwargs)
#   File "/home/c206/anaconda3/envs/M3DM/lib/python3.8/site-packages/huggingface_hub/file_download.py", line 1645, in get_hf_file_metadata
#     r = _request_wrapper(
#   File "/home/c206/anaconda3/envs/M3DM/lib/python3.8/site-packages/huggingface_hub/file_download.py", line 372, in _request_wrapper
#     response = _request_wrapper(
#   File "/home/c206/anaconda3/envs/M3DM/lib/python3.8/site-packages/huggingface_hub/file_download.py", line 395, in _request_wrapper
#     response = get_session().request(method=method, url=url, **params)
#   File "/home/c206/anaconda3/envs/M3DM/lib/python3.8/site-packages/requests/sessions.py", line 589, in request
#     resp = self.send(prep, **send_kwargs)
#   File "/home/c206/anaconda3/envs/M3DM/lib/python3.8/site-packages/requests/sessions.py", line 703, in send
#     r = adapter.send(request, **kwargs)
#   File "/home/c206/anaconda3/envs/M3DM/lib/python3.8/site-packages/huggingface_hub/utils/_http.py", line 66, in send
#     return super().send(request, *args, **kwargs)
#   File "/home/c206/anaconda3/envs/M3DM/lib/python3.8/site-packages/requests/adapters.py", line 622, in send
#     raise ConnectionError(e, request=request)
# requests.exceptions.ConnectionError: (MaxRetryError("HTTPSConnectionPool(host='huggingface.co', port=443): Max retries exceeded with url: /microsoft/resnet-50/resolve/main/preprocessor_config.json (Caused by NewConnectionError('<urllib3.connection.HTTPSConnection object at 0x7fb8c5bf4910>: Failed to establish a new connection: [Errno 101] Network is unreachable'))"), '(Request ID: 1112af18-ffb5-4ca9-a5df-8dca536dcbd8)')

# The above exception was the direct cause of the following exception:

# Traceback (most recent call last):
#   File "/home/c206/anaconda3/envs/M3DM/lib/python3.8/site-packages/transformers/utils/hub.py", line 399, in cached_file
#     resolved_file = hf_hub_download(
#   File "/home/c206/anaconda3/envs/M3DM/lib/python3.8/site-packages/huggingface_hub/utils/_validators.py", line 114, in _inner_fn
#     return fn(*args, **kwargs)
#   File "/home/c206/anaconda3/envs/M3DM/lib/python3.8/site-packages/huggingface_hub/file_download.py", line 1221, in hf_hub_download
#     return _hf_hub_download_to_cache_dir(
#   File "/home/c206/anaconda3/envs/M3DM/lib/python3.8/site-packages/huggingface_hub/file_download.py", line 1325, in _hf_hub_download_to_cache_dir
#     _raise_on_head_call_error(head_call_error, force_download, local_files_only)
#   File "/home/c206/anaconda3/envs/M3DM/lib/python3.8/site-packages/huggingface_hub/file_download.py", line 1826, in _raise_on_head_call_error
#     raise LocalEntryNotFoundError(
# huggingface_hub.utils._errors.LocalEntryNotFoundError: An error happened while trying to locate the file on the Hub and we cannot find the requested files in the local cache. Please check your connection and try again or make sure your Internet connection is on.

# The above exception was the direct cause of the following exception:

# Traceback (most recent call last):
#   File "timm-install-test.py", line 24, in <module>
#     processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
#   File "/home/c206/anaconda3/envs/M3DM/lib/python3.8/site-packages/transformers/models/auto/image_processing_auto.py", line 363, in from_pretrained
#     config_dict, _ = ImageProcessingMixin.get_image_processor_dict(pretrained_model_name_or_path, **kwargs)
#   File "/home/c206/anaconda3/envs/M3DM/lib/python3.8/site-packages/transformers/image_processing_utils.py", line 334, in get_image_processor_dict
#     resolved_image_processor_file = cached_file(
#   File "/home/c206/anaconda3/envs/M3DM/lib/python3.8/site-packages/transformers/utils/hub.py", line 442, in cached_file
#     raise EnvironmentError(
# OSError: We couldn't connect to 'https://huggingface.co' to load this file, couldn't find it in the cached files and it looks like microsoft/resnet-50 is not the path to a directory containing a file named preprocessor_config.json.
# Checkout your internet connection or see how to run the library in offline mode at 'https://huggingface.co/docs/transformers/installation#offline-mode'.
# (M3DM) (base) c206@c206linux:/data/wangwei/X-23d-Y-ai-Z-detection/misc-test-windows$ 
import timm
print(timm.models.create_model('tf_efficientnetv2_s').default_cfg)