
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab

import requests
from io import BytesIO
from PIL import Image
import numpy as np

pylab.rcParams['figure.figsize'] = 20, 12

from maskrcnn_benchmark.config import cfg
from predictor import COCODemo

# 参数配置文件
config_file ='/mnt/MSGAN/configs/e2e_mask_rcnn_R_101_FPN_1x_build.yaml'  #配置文件路径

# update the config options with the config file
cfg.merge_from_file(config_file)
# manual override some options
cfg.merge_from_list(["MODEL.DEVICE", "cuda"])
# cfg.MODEL.WEIGHT = '../pretrained/e2e_mask_rcnn_R_101_FPN_1x.pth'

coco_demo = COCODemo(cfg, min_image_size=800, confidence_threshold=0.7, )

# if False:
#     pass
# else:
#imgurl = "http://farm3.staticflickr.com/2469/3915380994_2e611b1779_z.jpg"
# response = requests.get(imgurl)
# pil_image = Image.open(BytesIO(response.content)).convert("RGB")

imgfile = '/mnt/MSGAN/datasets/build/images/val/01618.jpg'     #测试图片路径
# imgfile = '/mnt/MSGAN/datasets/build/images/train/JPEGImages'
pil_image = Image.open(imgfile)#.convert("RGB")
# pil_image.show()
image = np.array(pil_image)[:, :, [2, 1, 0]]
# print(image)
# forward predict
predictions = coco_demo.run_on_opencv_image(image)
# print(predictions)

# vis
plt.subplot(1, 2, 1)
plt.imshow(image[:, :, ::-1])
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(predictions[:, :, ::-1])
plt.axis('off')
plt.savefig("/mnt/MSGAN/datasets/build/test5.png")  #生成带有候选框的图片路径

# plt.savefig("/mnt/MSGAN/datasets/build/images/test")