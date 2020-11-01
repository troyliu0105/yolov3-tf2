import tensorflow as tf
from absl import flags
from absl.flags import FLAGS
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Lambda

from .backbones import build_backbone, backbone_list

__all__ = ['build_yolo_v3', 'yolo_nms', 'yolo_boxes']

flags.DEFINE_integer('yolo_max_boxes', 100,
                     'maximum number of boxes per image')
flags.DEFINE_float('yolo_iou_threshold', 0.5, 'iou threshold')
flags.DEFINE_float('yolo_score_threshold', 0.5, 'score threshold')
flags.DEFINE_enum('backbone', 'darknet',
                  backbone_list,
                  'choose backbone')


def yolo_boxes(pred, anchors, classes):
    # pred: (batch_size, grid, grid, anchors, (x, y, w, h, obj, ...classes))
    grid_size = tf.shape(pred)[1:3]
    box_xy, box_wh, objectness, class_probs = tf.split(pred, (2, 2, 1, classes), axis=-1)

    box_xy = tf.sigmoid(box_xy)
    objectness = tf.sigmoid(objectness)
    class_probs = tf.sigmoid(class_probs)
    pred_box = tf.concat((box_xy, box_wh), axis=-1)  # original xywh for loss

    # !!! grid[x][y] == (y, x)
    grid = tf.meshgrid(tf.range(grid_size[1]), tf.range(grid_size[0]))
    grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)  # [gx, gy, 1, 2]

    box_xy = (box_xy + tf.cast(grid, tf.float32)) / tf.cast(grid_size, tf.float32)
    box_wh = tf.exp(box_wh) * anchors

    box_x1y1 = box_xy - box_wh / 2
    box_x2y2 = box_xy + box_wh / 2
    bbox = tf.concat([box_x1y1, box_x2y2], axis=-1)

    return bbox, objectness, class_probs, pred_box


def yolo_nms(outputs, anchors, masks, classes):
    # boxes, conf, type
    b, c, t = [], [], []

    for o in outputs:
        b.append(tf.reshape(o[0], (tf.shape(o[0])[0], -1, tf.shape(o[0])[-1])))
        c.append(tf.reshape(o[1], (tf.shape(o[1])[0], -1, tf.shape(o[1])[-1])))
        t.append(tf.reshape(o[2], (tf.shape(o[2])[0], -1, tf.shape(o[2])[-1])))

    bbox = tf.concat(b, axis=1)
    confidence = tf.concat(c, axis=1)
    class_probs = tf.concat(t, axis=1)

    scores = confidence * class_probs
    boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
        boxes=tf.reshape(bbox, (tf.shape(bbox)[0], -1, 1, 4)),
        scores=tf.reshape(scores, (tf.shape(scores)[0], -1, tf.shape(scores)[-1])),
        max_output_size_per_class=FLAGS.yolo_max_boxes,
        max_total_size=FLAGS.yolo_max_boxes,
        iou_threshold=FLAGS.yolo_iou_threshold,
        score_threshold=FLAGS.yolo_score_threshold
    )

    return boxes, scores, classes, valid_detections


def yolo_v3(backbone_builder, anchors, masks, backbone_name, filters,
            size=None, channels=3, classes=80, training=False):
    x = inputs = Input([size, size, channels], name='input')

    model, yolo_conv, yolo_output = backbone_builder(name=f'yolo_{backbone_name}')

    outputs = []
    features = model(x)
    last = None

    for idx, (x, filters, mask) in enumerate(zip(features[::-1], filters, masks)):
        if last is not None:
            x = (last, x)
        x = yolo_conv(filters, name=f'yolo_conv_{idx}')(x)
        out = yolo_output(filters, len(mask), classes, name=f'yolo_output_{idx}')(x)
        outputs.append(out)
        last = x

    if training:
        return Model(inputs, outputs, name='yolov3')

    boxes = []
    for idx, (output, mask) in enumerate(zip(outputs, masks)):
        box = Lambda(lambda x: yolo_boxes(x, anchors[mask], classes),
                     name=f'yolo_boxes_{idx}')(output)
        boxes.append(box)

    outputs = Lambda(lambda x: yolo_nms(x, anchors, masks, classes),
                     name='yolo_nms')([box[:3] for box in boxes])

    return Model(inputs, outputs, name='yolov3')


def build_yolo_v3(backbone='darknet', size=None, channels=3, classes=80, training=False):
    fn, anchors, masks, filters = build_backbone(backbone)
    model = yolo_v3(fn, anchors, masks, backbone, filters,
                    size=size, channels=channels, classes=classes, training=training, )
    return model, anchors, masks
