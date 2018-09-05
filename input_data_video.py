
# coding: utf-8


import tensorflow as tf
import numpy as np

#np_mean = np.load('crop_mean.npy').reshape([16, 112, 112, 3])

def read_and_decode(filename_queue, vshape, normalize=False):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
        'video_raw_lr': tf.FixedLenFeature([], tf.string),
        'video_raw_hr': tf.FixedLenFeature([], tf.string),
        'label': tf.FixedLenFeature([], tf.int64),
    })

    video_lr = tf.decode_raw(features['video_raw_lr'], tf.uint8)
    video_hr = tf.decode_raw(features['video_raw_hr'], tf.uint8)
    label = tf.cast(features['label'], tf.int32)
    height = 112
    width = 112
    depth = 16
    video_lr = tf.reshape(video_lr, [depth, height, width, 3])
    #video = tf.random_crop(video, [depth, 112, 112, nchannel])
    video_lr = tf.cast(video_lr, tf.float32) / 255.0
    video_hr = tf.reshape(video_hr, [depth, height, width, 1])
    video_hr = tf.cast(video_hr, tf.float32) / 255.0
    #video = video - np_mean
    return video_lr, video_hr, label

def inputs(filenames, batch_size, num_epochs, num_threads, vshape, num_examples_per_epoch, shuffle=True):
    if not num_epochs:
        num_epochs = None
    for filename in filenames:
        if not tf.gfile.Exists(filename):
            raise ValueError('Failed to find file: ' + filename)
    with tf.name_scope('input'):
        filename_queue = tf.train.string_input_producer(
            filenames, num_epochs=num_epochs, shuffle=shuffle, name='string_input_producer'
        )
        video_lr, video_hr, label = read_and_decode(filename_queue, vshape, normalize=False)

        print('Video(LR) shape is: ', video_lr.get_shape())
        print('Video(HR) shape is: ', video_hr.get_shape())
        print('Label shape is ', label.get_shape())
        min_fraction_of_examples_in_queue = 0.4
        min_queue_examples = int(num_examples_per_epoch * min_fraction_of_examples_in_queue)
        if shuffle:
            videos_lr, videos_hr, sparse_labels = tf.train.shuffle_batch(
                [video_lr, video_hr, label],
                batch_size=batch_size,
                num_threads=num_threads,
                capacity=num_examples_per_epoch,
                enqueue_many=False,
                min_after_dequeue=min_queue_examples,
                name='batching_shuffling'
            )
        else:
            videos_lr, videos_hr, sparse_labels = tf.train.batch(
                [video_lr, video_hr, label],
                batch_size=batch_size,
                num_threads=num_threads,
                capacity=num_examples_per_epoch,
                enqueue_many=False,
                allow_smaller_final_batch=False,
                name='batching_shuffling'
            )
        print('Videos(LR) shape is ', videos_lr.get_shape())
        print('Videos(HR) shape is ', videos_hr.get_shape())
        print('Label shape is ', sparse_labels.get_shape())
        print('######################################################################')

    return videos_lr, videos_hr, sparse_labels
