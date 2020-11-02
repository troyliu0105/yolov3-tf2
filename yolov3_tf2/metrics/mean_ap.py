from typing import Tuple, Union

import numpy as np
import tensorflow as tf
from mean_average_precision import MetricBuilder
from tensorflow.keras.metrics import Metric


def postprocessing_nms_results(preds_raw: Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]):
    boxes, scores, classes, valid_detections = [p.numpy() for p in preds_raw]
    preds = []
    for batch_idx, valid in enumerate(valid_detections):
        pred_boxes = []
        for i in range(valid):
            b = boxes[batch_idx][i]
            c = classes[batch_idx][i]
            s = scores[batch_idx][i]
            box = np.concatenate((b, c, s))
            pred_boxes.append(box)
        pred_boxes = np.array(pred_boxes, dtype=np.float32)
        preds.append(pred_boxes)
    return preds


def postprocessing_y_labels(y_train: Union[tf.Tensor, np.ndarray]):
    """
    :param y_train: [B, 100, 4]
    :return:
    """
    if isinstance(y_train, tf.Tensor):
        y_train = y_train.numpy()
    gts = []
    for gt in y_train:
        gts.append(gt[~np.all(gt == 0, axis=1)])
    return gts


class MeanAP(Metric):
    def __init__(self, classes=20, iou_threshold=0.5, name='mAP', dtype=None, **kwargs):
        super(MeanAP, self).__init__(name=name, dtype=dtype, **kwargs)
        self.iou_threshold = iou_threshold
        self.num_classes = classes
        # self.class_names = class_names
        self.reset_states()
        self.metric_fn = MetricBuilder.build_evaluation_metric("map_2d", async_mode=False, num_classes=self.num_classes)

    def update_state(self, preds, gts, *args, **kwargs):
        preds = postprocessing_nms_results(preds)
        gts = postprocessing_y_labels(gts)
        assert len(preds) == len(gts)
        for p, g in zip(preds, gts):
            if g.shape[-1] != 7:
                pad = 7 - g.shape[-1]
                pad = np.zeros((g.shape[0], pad), dtype=g.dtype)
                g = np.concatenate((g, pad), axis=-1)
            self.metric_fn.add(p, g)

    def result(self):
        value = self.metric_fn.value(iou_thresholds=0.5)
        return value['mAP']
