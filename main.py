from model import SRCNN

import tensorflow as tf
import datetime

import pprint
import os
import scipy.io
import numpy as np
from tf_flags import FLAGS
os.environ['TF_CPP_MIN_LOG_LEVEL']="2"
#os.environ["CUDA_VISIBLE_DEVICES"]="4"
os.environ["CUDA_VISIBLE_DEVICES"]="0,3,4,5"
pp = pprint.PrettyPrinter()

def main(_):
    pp.pprint(FLAGS.__flags)

    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    if not os.path.exists(FLAGS.sample_dir):
        os.makedirs(FLAGS.sample_dir)

    model_dir = os.path.join(os.getcwd(), 'pretrained_models/9-1-5(91 images)')
    model1 = scipy.io.loadmat('{}/{}.mat'.format(model_dir, 'x3'))
    model1['weights_conv1'] = np.reshape(model1['weights_conv1'], (9, 9, 1, 64)).transpose(1, 0, 2, 3)
    model1['weights_conv2'] = np.reshape(model1['weights_conv2'].transpose(1,0,2), (1, 1, 64, 32))
    model1['weights_conv3'] = np.reshape(model1['weights_conv3'].transpose(1,0), (5, 5, 32, 1)).transpose(1, 0, 2, 3)

    for k, v in model1.items():
        if type(v) is np.ndarray:
            print(k, v.shape)
    model2 = scipy.io.loadmat('{}/{}.mat'.format(model_dir, 'x4'))
    model2['weights_conv1'] = np.reshape(model2['weights_conv1'], (9, 9, 1, 64)).transpose(1, 0, 2, 3)
    model2['weights_conv2'] = np.reshape(model2['weights_conv2'].transpose(1, 0, 2), (1, 1, 64, 32))
    model2['weights_conv3'] = np.reshape(model2['weights_conv3'].transpose(1, 0), (5, 5, 32, 1)).transpose(1, 0, 2, 3)

    for k, v in model2.items():
        if type(v) is np.ndarray:
            print(k, v.shape)
    srcnn = SRCNN(model1 = model1,
                  model2 = model2,
                  image_size=FLAGS.image_size,
                  label_size=FLAGS.label_size,
                  batch_size=FLAGS.batch_size,
                  c_dim=FLAGS.c_dim,
                  checkpoint_dir=FLAGS.checkpoint_dir,
                  sample_dir=FLAGS.sample_dir)
    
    #srcnn.train(FLAGS)
    srcnn.test_domain_adapted(FLAGS)
    #srcnn.test_single_GPU(FLAGS)
    #srcnn.test_multiple_GPU(FLAGS)
    #srcnn.test_multiple_CPU(FLAGS)

if __name__ == '__main__':
    tf.app.run()
