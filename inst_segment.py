import os
import sys
from mrcnn import utils
from mrcnn.config import Config
import mrcnn.model as modellib
import skimage.io
from skimage.color import rgba2rgb
import numpy as np


def segment_color_list(image_file, class_name):
    image = skimage.io.imread(image_file)
    if image.shape[2] == 4:
        image = rgba2rgb(image)
        image = np.round(image*255).astype(np.uint8)
    result, class_names = segmentation(image)
    class_id = class_names.index(class_name)
    class_ids = list(result['class_ids'])

    if class_id in class_ids:
        class_index = class_ids.index(class_id)

        mask = []
        for row in result['masks']:
            mask.append([[i[class_index]] for i in row])
        mask = np.array(mask)

        color_list = []
        for row, bools in list(zip(image, mask)):
            for i in row[bools.flatten()].tolist():
                color_list.append(i)

        inst_info = (
            image,
            result['rois'][class_index: class_index+1],
            mask,
            result['class_ids'][class_index: class_index+1],
            class_names
            )

        return color_list, inst_info

    else:
        print(f'{class_name} is not detected')
        sys.exit()


class CocoConfig(Config):
    NAME = "coco"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 80


def segmentation(image):
    ROOT_DIR = os.path.abspath("./")
    MODEL_DIR = os.path.join(ROOT_DIR, "logs")
    COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

    if not os.path.exists(COCO_MODEL_PATH):
        utils.download_trained_weights(COCO_MODEL_PATH)

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

    results = model.detect([image], verbose=1)
    r = results[0]

    return r, class_names
