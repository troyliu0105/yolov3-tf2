import numpy as np
import tensorflow as tf
from absl import app, flags, logging
from absl.flags import FLAGS

from yolov3_tf2.models import yolo_v3, yolo_v3_tiny
from yolov3_tf2.utils import load_darknet_weights

flags.DEFINE_string('weights', './data/yolov3.weights', 'path to weights file')
flags.DEFINE_string('output', './checkpoints/yolov3.tf', 'path to output')
flags.DEFINE_boolean('tiny', False, 'yolov3 or yolov3-tiny')
flags.DEFINE_integer('num_classes', 80, 'number of classes in the model')


def main(_argv):
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    if FLAGS.tiny:
        yolo = yolo_v3_tiny(classes=FLAGS.num_classes)
    else:
        yolo = yolo_v3(classes=FLAGS.num_classes)
    yolo.summary()
    logging.info('model created')

    load_darknet_weights(yolo, FLAGS.weights, FLAGS.tiny)
    logging.info('weights loaded')

    img = np.random.random((1, 320, 320, 3)).astype(np.float32)
    output = yolo(img)
    logging.info('sanity check passed')

    yolo.save_weights(FLAGS.output)
    logging.info('weights saved')


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
