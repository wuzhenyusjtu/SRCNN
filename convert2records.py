
# coding: utf-8

# In[8]:


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import glob
import tensorflow as tf
import numpy as np
import h5py
from tf_flags import FLAGS
from utils import preprocess_py
from tqdm import tqdm



def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _floats_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


# In[2]:


def convert_to_images(images, labels, name, directory):
    if images.shape[0] != labels.shape[0]:
        raise ValueError('Images size %d does not match labels size %d.' %
                         (images.shape[0], labels.shape[0]))
    num_examples = images.shape[0]
    height = images.shape[1]
    width = images.shape[2]
    nchannel = images.shape[3]

    filename = os.path.join(directory, '{}.tfrecords'.format(name))
    print('Writing', filename)
    writer = tf.python_io.TFRecordWriter(filename)
    for index in range(num_examples):
        image_raw = images[index].astype(np.float32).tostring()
        label_raw = labels[index].astype(np.float32).tostring()
        example = tf.train.Example(features=tf.train.Features(feature={
            'height': _int64_feature(height),
            'width': _int64_feature(width),
            'nchannel': _int64_feature(nchannel),
            'label_raw': _bytes_feature(label_raw),
            'image_raw': _bytes_feature(image_raw)
        }))
        writer.write(example.SerializeToString())
    writer.close()

def convert_to_videos(videos, labels, name, directory):
    if videos.shape[0] != labels.shape[0]:
        raise ValueError('Videos size %d does not match labels size %d.' %
                             (videos.shape[0], labels.shape[0]))

    num_examples = videos.shape[0]

    # videos = videos.transpose(0, 3, 1, 2, 4)

    filename = os.path.join(directory, '{}.tfrecords'.format(name))
    print('Writing', filename)
    writer = tf.python_io.TFRecordWriter(filename)
    for index in range(num_examples):
        video_raw = videos[index].tostring()
        example = tf.train.Example(features=tf.train.Features(feature={
            'label': _int64_feature(int(labels[index])),
            'video_raw': _bytes_feature(video_raw),
        }))
        writer.write(example.SerializeToString())
    writer.close()

# In[11]:


def prepare_data(dataset):
    """
    Args:
    dataset: choose train dataset or test dataset

    For train dataset, output data would be ['.../t1.bmp', '.../t2.bmp', ..., '.../t99.bmp']
    """
    if dataset == "Train":
        data_dir = os.path.join(os.getcwd(), dataset)
        data = glob.glob(os.path.join(data_dir, "*.bmp"))
    else:
        data_dir = os.path.join(os.sep, (os.path.join(os.getcwd(), "Test")), "Set5")
        #data_dir = os.path.join(os.getcwd(), dataset)
        data = glob.glob(os.path.join(data_dir, "*.bmp"))
        print(data)

    return data

def get_subimages(data, config, dataset):
    sub_input_sequence = []
    sub_label_sequence = []
    padding = abs(config.image_size - config.label_size) // 2 # 6

    if dataset == "Train":
        stride = config.tr_stride
    else:
        stride = config.val_stride

    image_size = config.image_size
    label_size = config.label_size
    pbar = tqdm(total=len(data), )
    for i in range(len(data)):
        pbar.update(1)
        input_, label_ = preprocess_py(data[i], config.scale)

        if len(input_.shape) == 3:
            h, w, _ = input_.shape
        else:
            h, w = input_.shape

        for x in range(0, h-image_size+1, stride):
            for y in range(0, w-image_size+1, stride):
                sub_input = input_[x:x+image_size, y:y+image_size] # [33 x 33]
                sub_label = label_[x+padding:x+padding+label_size, y+padding:y+padding+label_size] # [21 x 21]

                # Make channel value
                sub_input = sub_input.reshape([image_size, image_size, 1])
                sub_label = sub_label.reshape([label_size, label_size, 1])

                sub_input_sequence.append(sub_input)
                sub_label_sequence.append(sub_label)
    pbar.close()

    """
    len(sub_input_sequence) : the number of sub_input (33 x 33 x ch) in one image
    (sub_input_sequence[0]).shape : (33, 33, 1)
    """
    # Make list to numpy array. With this transform
    arrdata = np.asarray(sub_input_sequence) # [?, 33, 33, 1]
    arrlabel = np.asarray(sub_label_sequence) # [?, 21, 21, 1]
    print(arrdata.shape)
    print(arrlabel.shape)
    return arrdata, arrlabel


# In[12]:

if __name__ == "__main__":
    savepath = os.path.join(os.getcwd(), 'checkpoint')
    for dataset in ["Train", "Test"]:
        data = prepare_data(dataset)
        arrdata, arrlabel = get_subimages(data, FLAGS, dataset)
        convert_to_images(arrdata, arrlabel, name=dataset, directory=savepath)
