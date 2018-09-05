
# coding: utf-8

# In[ ]:


# uncompyle6 version 2.13.2
# Python bytecode 3.6 (3379)
# Decompiled from: Python 3.6.1 |Anaconda 4.4.0 (64-bit)| (default, May 11 2017, 13:09:58)
# [GCC 4.4.7 20120313 (Red Hat 4.4.7-1)]
# Embedded file name: /home/wuzhenyu_sjtu/Desktop/C3D/CPU_only/VideoReader.py
# Compiled at: 2017-10-16 13:58:04
# Size of source mod 2**32: 4912 bytes
import numpy as np
import cv2
import os
from tqdm import tqdm
import random
from random import randint

class VideoReader:

    def __init__(self, height, widthr):
        self.width = width
        self.height = height
        self.train_set = set()
        self.test_set = set()
        self.read_train_test_split('ucfTrainTestlist/trainlist01.txt', 'ucfTrainTestlist/testlist01.txt')

    def read_train_test_split(self, filename_train, filename_test):
        with open(filename_train, 'r') as f:
            for line in f:
                self.train_set.add(line.strip('\n').split()[0].split('/')[-1])
        with open(filename_test, 'r') as f:
            for line in f:
                self.test_set.add(line.strip('\n').split()[0].split('/')[-1])


    def read(self, filename):
        cap = cv2.VideoCapture(filename)
        nframe = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, randint(0, nframe-1))
        ret, frame = cap.read()
        # Convert to RGB space
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_CUBIC)
        cap.release()
        return frame

    def loaddata(self, video_dir):
        filenames = []
        for path, subdirs, files in os.walk(video_dir):
            for name in files:
                filenames.append(os.path.join(path, name))
        pbar = tqdm(total=len(filenames),)
        frames_train = []
        frames_test = []
        for filename in filenames:
            pbar.update(1)
            print(filename)
            try:
                fname = filename.split('/')[-1]
                if fname in self.train_set:
                    frame = self.read(filename)
                    frames_train.append(frame)
                elif fname in self.test_set:
                    frame = self.read(filename)
                    frames_test.append(frame)
                else:
                    print('{} not found in train/test split'.format(fname))
                    continue
            except cv2.error  as e:
                print(e)
                print('{} is unable to read'.format(filename))
                pass
        pbar.close()
        return frames_train, frames_test

if __name__ == "__main__":
    height, width = 112, 112
    vreader = VideoReader(height, width)
    videos_dir = 'UCF-101'
    frames_train, frames_test = vreader.loaddata(videos_dir)
    for i in range(len(frames_train)):
        frame = cv2.cvtColor(frames_train[i], cv2.COLOR_RGB2BGR)
        cv2.imwrite('UCF-frames/Train/t{}.bmp'.format(i), frame)
    for i in range(len(frames_test)):
        frame = cv2.cvtColor(frames_test[i], cv2.COLOR_RGB2BGR)
        cv2.imwrite('UCF-frames/Test/t{}.bmp'.format(i), frame)
