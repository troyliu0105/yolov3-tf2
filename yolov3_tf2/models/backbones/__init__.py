import numpy as np

from . import darknet
from . import mobilenet

yolo_anchors = np.array([(10, 13), (16, 30), (33, 23),
                         (30, 61), (62, 45), (59, 119),
                         (116, 90), (156, 198), (373, 326)],
                        np.float32) / 416
yolo_anchor_masks = np.array([[6, 7, 8], [3, 4, 5], [0, 1, 2]])

yolo_tiny_anchors = np.array([(10, 14), (23, 27), (37, 58),
                              (81, 82), (135, 169), (344, 319)],
                             np.float32) / 416
yolo_tiny_anchor_masks = np.array([[3, 4, 5], [0, 1, 2]])

__CONFIGS = {
    'darknet': {
        'fn': darknet.darknet,
        'anchors': yolo_anchors,
        'masks': yolo_anchor_masks,
        'filters': [512, 256, 128]
    },
    'darknet-tiny': {
        'fn': darknet.darknet_tiny,
        'anchors': yolo_tiny_anchors,
        'masks': yolo_tiny_anchor_masks,
        'filters': [256, 128]
    },
    'mobilenetv1': {
        'fn': mobilenet.mobilenet_v1,
        'anchors': yolo_anchors,
        'masks': yolo_anchor_masks,
        'filters': [512, 256, 128]
    },
    'mobilenetv2': {
        'fn': mobilenet.mobilenet_v2,
        'anchors': yolo_anchors,
        'masks': yolo_anchor_masks,
        'filters': [512, 256, 128]
    },
    'mobilenetv3_large': {
        'fn': mobilenet.mobilenet_v3_large,
        'anchors': yolo_anchors,
        'masks': yolo_anchor_masks,
        'filters': [512, 256, 128]
    },
}
backbone_list = list(__CONFIGS.keys())


def build_backbone(backbone_name):
    config = __CONFIGS[backbone_name]
    fn = config['fn']
    anchors = config['anchors']
    masks = config['masks']
    filters = config['filters']
    return fn, anchors, masks, filters
