
# coding: utf-8

# In[ ]:


# coding: utf-8

import os
import numpy as np
import tensorflow as tf


def read_and_decode(filename_queue, image_shape, label_shape):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
        'image_raw': tf.FixedLenFeature([], tf.string),
        'label_raw': tf.FixedLenFeature([], tf.string)
    })

    image = tf.decode_raw(features['image_raw'], tf.float32)
    label = tf.decode_raw(features['label_raw'], tf.float32)
    image = tf.reshape(image, image_shape)
    image.set_shape(image_shape)
    label = tf.reshape(label, label_shape)
    label.set_shape(label_shape)

    return image, label

def inputs(filename, batch_size, num_epochs, num_threads, image_shape, label_shape, num_examples_per_epoch):
    if not num_epochs:
        num_epochs = None
    if not tf.gfile.Exists(filename):
        raise ValueError('Failed to find file: ' + filename)
    with tf.name_scope('input'):
        filename_queue = tf.train.string_input_producer(
            [filename], num_epochs=num_epochs, name='string_input_producer'
        )
        image, label = read_and_decode(filename_queue, image_shape, label_shape)

        print('Image shape is ', image.get_shape())
        print('Label shape is ', label.get_shape())
        print('################################################################################################################')

        min_fraction_of_examples_in_queue = 0.4
        min_queue_examples = int(num_examples_per_epoch * min_fraction_of_examples_in_queue)

        images, sparse_labels = tf.train.shuffle_batch(
            [image, label],
            batch_size=batch_size,
            num_threads=num_threads,
            capacity=min_queue_examples + 3 * batch_size,
            enqueue_many=False,
            min_after_dequeue=min_queue_examples,
            name='batching_shuffling'
        )
    return images, sparse_labels
