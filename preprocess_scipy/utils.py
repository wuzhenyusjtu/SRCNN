"""
Scipy version > 0.18 is needed, due to 'mode' option from scipy.misc.imread function
"""

import os
import glob
import random
import matplotlib.pyplot as plt

from PIL import Image  # for loading images as YCbCr format
import scipy.misc
import scipy.ndimage
import numpy as np
import cv2
import tensorflow as tf
try:
    import matlab.engine
    MATLAB_ENGINE = matlab.engine.start_matlab("-nojvm")
except ImportError as err:
    print(err)

def rgb2luma(im_rgb):
    #ycbcr = origT * rgb + origOffset;
    im_rgb = im_rgb / 255
    coef = (65.481, 128.553, 24.966)
    offset = 16;
    r, g, b = im_rgb[:,:,0], im_rgb[:,:,1], im_rgb[:,:,2]
    return r * coef[0] + g * coef[1] + b * coef[2] + offset

def preprocess(path, scale=3, use_matlab=False):
    """
    Preprocess single image file
    (1) Read original image as YCbCr format (and grayscale as default)
    (2) Normalize
    (3) Apply image file with bicubic interpolation

    Args:
    path: file path of desired file
    input_: image applied bicubic interpolation (low-resolution)
    label_: image with original resolution (high-resolution)
    """
    if use_matlab:
        im = MATLAB_ENGINE.imread(path)
        im = np.asarray(MATLAB_ENGINE.rgb2ycbcr(im))
        label_ = modcrop(im[:,:,0], scale)
        label_ = label_ / 255.
        input_ = matlab.single(label_.tolist())
        input_ = MATLAB_ENGINE.imresize(input_, 1./scale, 'bicubic')

        input_ = MATLAB_ENGINE.imresize(input_, scale, 'bicubic')
        input_ = np.array(input_)

    else:
        im = cv2.imread(path, cv2.IMREAD_COLOR)
        im = rgb2luma(im)
        #im = scipy.misc.imread(path, mode='YCbCr').astype(np.float)
        label_ = modcrop(im, scale)
        #Must be normalized
        label_ = label_ / 255.
        input_ = cv2.resize(label_, None, fx=1./scale, fy=1./scale, interpolation=cv2.INTER_CUBIC)
        input_ = cv2.resize(input_, None, fx=scale/1., fy=scale/1., interpolation=cv2.INTER_CUBIC)
        input_ = scipy.ndimage.interpolation.zoom(label_, (1./scale), prefilter=False)
        input_ = scipy.ndimage.interpolation.zoom(input_, (scale/1.), prefilter=False)

    return input_, label_


def modcrop(image, scale=3):
    """
    To scale down and up the original image, first thing to do is to have no remainder while scaling operation.

    We need to find modulo of height (and width) and scale factor.
    Then, subtract the modulo from height (and width) of original image size.
    There would be no remainder even after scaling operation.
    """
    if len(image.shape) == 3:
        h, w, _ = image.shape
        h = h - np.mod(h, scale)
        w = w - np.mod(w, scale)
        image = image[0:h, 0:w, :]
    else:
        h, w = image.shape
        h = h - np.mod(h, scale)
        w = w - np.mod(w, scale)
        image = image[0:h, 0:w]
    return image

def shave(image, border):
    if len(image.shape) == 3:
        h, w, _ = image.shape
        return image[border[0]:h-border[1], border[0]:w-border[1], :]
    else:
        h, w = image.shape
        return image[border[0]:h-border[1], border[0]:w-border[1]]

def test_setup(config):
    data_dir = os.path.join(os.sep, (os.path.join(os.getcwd(), "Test")), "Set14")
    data = glob.glob(os.path.join(data_dir, "*.bmp"))
    input_sequence = []
    label_sequence = []
    for i in range(len(data)):
        input_, label_ = preprocess(data[i], config.scale)
        input_ = input_.reshape(input_.shape[0], input_.shape[1], 1)
        label_ = label_.reshape(label_.shape[0], label_.shape[1], 1)
        input_sequence.append(input_)
        label_sequence.append(label_)
        print(input_.shape)
        print(label_.shape)

    arrdata = np.asarray(input_sequence, dtype=object)
    arrlabel = np.asarray(label_sequence, dtype=object)
    return arrdata, arrlabel


def imsave(image, path):
    return cv2.imwrite(path, image)
