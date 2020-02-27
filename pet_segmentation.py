import os
import sys
import numpy as np
import skimage.io
from mrcnn import utils
from mrcnn.config import Config
import mrcnn.model as modellib
from PIL import Image


ROOT_DIR = os.path.abspath("./")
IMAGE_DIR = os.path.join(ROOT_DIR, "images")
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
OUTPUT_DIR = os.path.join(ROOT_DIR, "output/")

if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)


class CocoConfig(Config):
    NAME = "coco"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 80


config = CocoConfig()

model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
model.load_weights(COCO_MODEL_PATH, by_name=True)

class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']


file_name = next(os.walk(IMAGE_DIR))[2]
image = skimage.io.imread(os.path.join(IMAGE_DIR, file_name[0]))

results = model.detect([image], verbose=1)
r = results[0]

dog_id = class_names.index('dog')
class_ids = list(r['class_ids'])

if dog_id in class_ids:
    dog_index = class_ids.index(dog_id)

print(r['masks'][0])
mask = []
for row in r['masks']:
    mask.append([i[dog_index] for i in row])

color_list = []
for row, bools in list(zip(image, mask)):
    for i in row[bools].tolist():
        color_list.append(i)

height = len(color_list)//100
color_list = np.array(color_list[:(height*100)])
color_mat = color_list.reshape(height,100,3)

img = Image.fromarray(color_mat.astype('uint8'), 'RGB')
img.save(os.path.join(OUTPUT_DIR, "mid.png"))


