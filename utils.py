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
from matlab_imresize import imresize
from skimage.io import imread
try:
    import matlab.engine
    MATLAB_ENGINE = matlab.engine.start_matlab("-nojvm")
except ImportError as err:
    print(err)

def rgb2luma(im_rgb):
    #ycbcr = origT * rgb + origOffset;
    im_rgb = im_rgb / 255
    coef = (65.481, 128.553, 24.966)
    offset = 16
    r, g, b = im_rgb[:,:,0], im_rgb[:,:,1], im_rgb[:,:,2]
    return r * coef[0] + g * coef[1] + b * coef[2] + offset

def ycbcr2rgb(ycbcrs):
    ycbcrs = ycbcrs.copy()
    Tinv = np.asarray([[0.00456621, 0., 0.00625893],
                           [0.00456621, -0.00153632, -0.00318811],
                           [0.00456621, 0.00791071, 0.]])
    offset = np.asarray([16, 128, 128])
    ycbcrs[:, :, :, 0] -= offset[0]
    ycbcrs[:, :, :, 1] -= offset[1]
    ycbcrs[:, :, :, 2] -= offset[2]

    rgbs = []
    for i in range(ycbcrs.shape[0]):
        ycbcr = ycbcrs[i]
        rgb = (np.dot(ycbcr, Tinv.T) * 255.0).astype(np.uint8)
        rgbs.append(rgb)
    return np.asarray(rgbs).reshape(ycbcrs.shape[0], ycbcrs.shape[1], ycbcrs.shape[2], ycbcrs.shape[3])

def preprocess_py(path, scale=3):
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

    im_uint8 = imread(path)
    im_double = rgb2luma(im_uint8) / 255.

    label_ = modcrop(im_double, scale)
    input_ = imresize(label_, scalar_scale=1./scale)
    #print(input_.shape)
    input_ = imresize(input_, scalar_scale=scale)
    #print(input_.shape)

    return input_, label_

def preprocess_mat(path, scale=3):
    im = MATLAB_ENGINE.imread(path)
    im = np.asarray(MATLAB_ENGINE.rgb2ycbcr(im))
    label_ = modcrop(im[:,:,0], scale) / 255.
    input_ = matlab.single(label_.tolist())
    input_ = MATLAB_ENGINE.imresize(input_, 1./scale, 'bicubic')
    input_ = MATLAB_ENGINE.imresize(input_, scale, 'bicubic')
    input_ = np.asarray(input_)
    return input_, label_

def preprocess_cv2(path, scale=3):
    im = cv2.imread(path, cv2.IMREAD_COLOR).astype(np.uint8)[...,::-1]
    im = rgb2luma(im)
    label_ = modcrop(im, scale) / 255.
    input_ = cv2.resize(label_, None, fx=1./scale, fy=1./scale, interpolation=cv2.INTER_CUBIC)
    input_ = cv2.resize(input_, None, fx=scale/1., fy=scale/1., interpolation=cv2.INTER_CUBIC)
    #input_ = scipy.ndimage.interpolation.zoom(label_, (1./scale), prefilter=False)
    #input_ = scipy.ndimage.interpolation.zoom(input_, (scale/1.), prefilter=False)
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
        if imread(data[i]).ndim != 3:
            continue
        input_, label_ = preprocess_py(data[i], config.scale)
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

def _prepend_edge(tensor, pad_amt, axis=1):
    '''
    This function is intented to add 'reflective' padding to a 4d Tensor across
        the height and width dimensions
    Parameters
    ----------
    tensor: Tensor with rank 4
    pad_amt: Integer
    axis: Integer
        Must be in (1,2)
    '''
    if axis not in (1, 2):
        raise ValueError("Axis must equal 0 or 1. Axis is set to %i" % axis)

    if axis == 1:
        concat_dim = 2
    else:
        concat_dim = 1

    begin = [0, 0, 0, 0]
    end = [-1, -1, -1, -1]
    end[axis] = 1

    edges = pad_amt*[tf.slice(tensor,begin,end)]
    if len(edges) > 1:
        padding = tf.concat(edges, axis)
    else:
        padding = edges[0]

    tensor_padded = tf.concat([padding, tensor],axis)
    return tensor_padded

def _append_edge(tensor, pad_amt, axis=1):
    '''
    This function is intented to add 'reflective' padding to a 4d Tensor across
        the height and width dimensions
    Parameters
    ----------
    tensor: Tensor with rank 4
    pad_amt: Integer
    axis: Integer
        Must be in (1,2)
    '''
    if axis not in (1, 2):
        raise ValueError("Axis must equal 0 or 1. Axis is set to %i" % axis)

    if axis == 1:
        concat_dim = 2
    else:
        concat_dim = 1

    begin = [0, 0, 0, 0]
    end = [-1, -1, -1, -1]
    begin[axis] = tf.shape(tensor)[axis]-1 # go to the end

    edges = pad_amt*[tf.slice(tensor,begin,end)]

    if len(edges) > 1:
        padding = tf.concat(edges, axis)
    else:
        padding = edges[0]

    tensor_padded = tf.concat([tensor, padding], axis)
    return tensor_padded

def replicate_padding(tensor, pad_amt):
    if isinstance(pad_amt, int):
        pad_amt = [pad_amt] * 2
    for axis, p in enumerate(pad_amt):
        tensor = _prepend_edge(tensor, p, axis=axis+1)
        tensor = _append_edge(tensor, p, axis=axis+1)
    return tensor

def write_video(videos, labels):
    # def write_video(videos, labels, sigmas):
    class_dict = {}
    with open('ucfTrainTestlist/classInd.txt', 'r') as f:
        for line in f:
            # print(line)
            words = line.strip('\n').split()
            class_dict[int(words[0]) - 1] = words[1]

    width, height = 112, 112
    for i in range(len(videos)):
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        output = '{}_{:d}.avi'.format(class_dict[labels[i]], i)
        # output = '{}_{:f}_{:d}.avi'.format(class_dict[labels[i]], sigmas[i], i)
        out = cv2.VideoWriter(output, fourcc, 1.0, (width, height), True)
        #vid = self.ycbcr2rgb(videos[i]*255)
        vid = videos[i].astype('uint8')
        for i in range(vid.shape[0]):
            frame = vid[i]
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            frame = frame.reshape(112, 112, 3)
            out.write(frame)
        out.release()
        cv2.destroyAllWindows()
