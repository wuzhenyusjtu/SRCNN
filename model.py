
# coding: utf-8

# In[ ]:



# coding: utf-8
from utils import (
    test_setup,
    imsave,
    shave,
    replicate_padding,
)

import time
import os
import matplotlib.pyplot as plt

import numpy as np
import tensorflow as tf
import input_data_image as input_data
import input_data_video
import h5py
import cv2
from convert2records import convert_to_videos
from utils import write_video
from utils import ycbcr2rgb
import gc
import resource


class SRCNN(object):

    def __init__(self,
                 model1,
                 model2,
                 image_size=33,
                 label_size=21,
                 batch_size=128,
                 c_dim=1,
                 checkpoint_dir=None,
                 sample_dir=None):
        self.model1 = model1
        self.model2 = model2
        self.is_grayscale = (c_dim == 1)
        self.image_size = image_size
        self.label_size = label_size
        self.batch_size = batch_size

        self.c_dim = c_dim

        self.checkpoint_dir = checkpoint_dir
        self.sample_dir = sample_dir

    def tower_loss(self, name_scope, preds, labels):
        #labels = tf.cast(labels, tf.int64)
        MSE = tf.reduce_mean(
            tf.square(labels - preds))
        tf.summary.scalar(
            name_scope + 'mse',
            MSE)
        return MSE

    def compute_psnr(self, tensor0, tensor1):
        MSE = tf.reduce_mean(
            tf.square(tensor0 - tensor1))
        psnr = tf.multiply(tf.constant(20, dtype=tf.float32),
                           tf.log(1 / tf.sqrt(MSE)) / tf.log(tf.constant(10, dtype=tf.float32)), name='psnr')
        return psnr

    def average_gradients(self, tower_grads):
        average_grads = []
        for grad_and_vars in zip(*tower_grads):
            # Note that each grad_and_vars looks like the following:
            #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
            grads = []
            for g, _ in grad_and_vars:
                # Add 0 dimension to the gradients to represent the tower.
                expanded_g = tf.expand_dims(g, 0)

                # Append on a 'tower' dimension which we will average over below.
                grads.append(expanded_g)

            # Average over the 'tower' dimension.
            grad = tf.concat(grads, 0)
            grad = tf.reduce_mean(grad, 0)

            # Keep in mind that the Variables are redundant because they are shared
            # across towers. So .. we will just return the first tower's pointer to
            # the Variable.
            v = grad_and_vars[0][1]
            grad_and_var = (grad, v)
            average_grads.append(grad_and_var)
        return average_grads

    def _variable_on_cpu(self, name, shape, initializer):
        with tf.device('/cpu:0'):
            var = tf.get_variable(name, shape, initializer=initializer)
        return var

    def _maybe_pad_x(self, x, padding, is_training):
        print(padding)
        if padding == 0:
            x_pad = x
        elif padding > 0:
            x_pad = tf.cond(is_training, lambda: x, lambda: replicate_padding(x, padding))
        else:
            raise ValueError("Padding value %i should be greater than or equal to 1" % padding)
        return x_pad

    def inference(self, X, isTrain=True):
        if isinstance(isTrain, bool):
            isTrain = tf.constant(isTrain)
        weights = {
            'w1': self._variable_on_cpu('w1', [9, 9, 1, 64], tf.constant_initializer(self.model1['weights_conv1'])),
            'w2': self._variable_on_cpu('w2', [1, 1, 64, 32], tf.constant_initializer(self.model1['weights_conv2'])),
            'w3': self._variable_on_cpu('w3', [5, 5, 32, 1], tf.constant_initializer(self.model1['weights_conv3'])),
        }

        biases = {
            'b1': self._variable_on_cpu('b1', [64], tf.constant_initializer(np.squeeze(self.model1['biases_conv1']))),
            'b2': self._variable_on_cpu('b2', [32], tf.constant_initializer(np.squeeze(self.model1['biases_conv2']))),
            'b3': self._variable_on_cpu('b3', [1], tf.constant_initializer(np.squeeze(self.model1['biases_conv3']))),
        }

        with tf.variable_scope('conv1') as scope:
            X = self._maybe_pad_x(X, (9-1)//2, isTrain)
            conv1 = tf.nn.relu(tf.nn.conv2d(X, weights['w1'], strides=[1,1,1,1], padding='VALID') + biases['b1'], name=scope.name)
            #conv1 = tf.nn.max_pool(conv1, ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME', name='pool1')
        with tf.variable_scope('conv2') as scope:
            conv2 = tf.nn.relu(tf.nn.conv2d(conv1, weights['w2'], strides=[1,1,1,1], padding='VALID') + biases['b2'], name=scope.name)
            #conv2 = tf.nn.max_pool(conv2, ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME', name='pool2')
        with tf.variable_scope('conv3') as scope:
            conv2 = self._maybe_pad_x(conv2, (5-1)//2, isTrain)
            conv3 = tf.add(tf.nn.conv2d(conv2, weights['w3'], strides=[1,1,1,1], padding='VALID'), biases['b3'], name=scope.name)
            conv3 = tf.nn.max_pool(conv3, ksize=[1,4,4,1], strides=[1,4,4,1], padding='SAME', name='pool1')

        return conv3

    def train(self, FLAGS):

        #images_placeholder = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, self.c_dim], name='images')
        #labels_placeholder = tf.placeholder(tf.float32, [None, self.label_size, self.label_size, self.c_dim], name='labels')
        images_placeholder = tf.placeholder(tf.float32, [None, None, None, self.c_dim], name='images')
        labels_placeholder = tf.placeholder(tf.float32, [None, None, None, self.c_dim], name='labels')


        tower_grads1 = []
        tower_grads2 = []
        losses = []
        #opt = tf.train.GradientDescentOptimizer(FLAGS.learning_rate)
        opt1 = tf.train.MomentumOptimizer(FLAGS.learning_rate, FLAGS.momentum)
        opt2 = tf.train.MomentumOptimizer(FLAGS.learning_rate * 0.1, FLAGS.momentum)

        with tf.variable_scope(tf.get_variable_scope()) as scope:
            for gpu_index in range(0, FLAGS.gpu_num):
                with tf.device('/gpu:%d' % gpu_index):
                    print('/gpu:%d' % gpu_index)
                    with tf.name_scope('%s_%d' % ('gpu', gpu_index)) as scope:
                        pred = self.inference(images_placeholder[gpu_index * FLAGS.batch_size:(gpu_index + 1) * FLAGS.batch_size])
                        loss = self.tower_loss(scope,
                                          pred,
                                          labels_placeholder[gpu_index * FLAGS.batch_size:(gpu_index + 1) * FLAGS.batch_size])
                        losses.append(loss)
                        grads = opt1.compute_gradients(loss)
                        grads1 = [v for v in grads if "conv3" not in v[0].op.name]
                        grads2 = [v for v in grads if "conv3" in v[0].op.name]
                        #grads = opt.compute_gradients(loss)
                        tower_grads1.append(grads1)
                        tower_grads2.append(grads2)
                        tf.get_variable_scope().reuse_variables()
        loss_op = tf.reduce_mean(losses, name='mse')
        psnr_op = tf.multiply(tf.constant(20, dtype=tf.float32), tf.log(1/tf.sqrt(loss_op))/tf.log(tf.constant(10, dtype=tf.float32)), name='psnr')
        tf.summary.scalar('loss', loss_op)
        grads1 = self.average_gradients(tower_grads1)
        grads2 = self.average_gradients(tower_grads2)
        apply_gradient_op1 = opt1.apply_gradients(grads1)
        apply_gradient_op2 = opt2.apply_gradients(grads2)
        # Stochastic gradient descent with the standard backpropagation
        train_op = tf.group(apply_gradient_op1, apply_gradient_op2)

        self.saver = tf.train.Saver(tf.global_variables())

        tr_images_op, tr_labels_op = input_data.inputs(filename = 'checkpoint/SRx8/Train.tfrecords',
                                                 batch_size=FLAGS.batch_size * FLAGS.gpu_num,
                                                 num_epochs=None,
                                                 num_threads=FLAGS.num_threads,
                                                 image_shape=[FLAGS.image_size,FLAGS.image_size,1],
                                                 label_shape=[FLAGS.label_size,FLAGS.label_size,1],
                                                 num_examples_per_epoch=FLAGS.num_examples_per_epoch)
        val_images_op, val_labels_op = input_data.inputs(filename = 'checkpoint/SRx8/Test.tfrecords',
                                                   batch_size=FLAGS.batch_size * FLAGS.gpu_num,
                                                   num_epochs=None,
                                                   num_threads=FLAGS.num_threads,
                                                   image_shape=[FLAGS.image_size,FLAGS.image_size,1],
                                                   label_shape=[FLAGS.label_size,FLAGS.label_size,1],
                                                   num_examples_per_epoch=FLAGS.num_examples_per_epoch)

        conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        conf.gpu_options.allow_growth = True
        sess = tf.Session(config=conf)
        init_op = tf.group(tf.local_variables_initializer(), tf.global_variables_initializer())
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)


        if self.load(sess, self.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")


        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(FLAGS.log_dir+'train', sess.graph)
        val_writer = tf.summary.FileWriter(FLAGS.log_dir+'val', sess.graph)
        print("Training...")

        for step in range(FLAGS.max_steps):
            # Run by batch images
            start_time = time.time()
            tr_images, tr_labels = sess.run([tr_images_op, tr_labels_op])
            tr_feed = {images_placeholder: tr_images, labels_placeholder: tr_labels}
            _, loss_value = sess.run([train_op, loss_op], feed_dict=tr_feed)

            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

            print("Step: [%2d], time: [%4.4f], training_loss = [%.8f]" % (step, time.time()-start_time, loss_value))

            if step % FLAGS.val_step == 0:
                val_images, val_labels = sess.run([val_images_op, val_labels_op])
                val_feed = {images_placeholder: val_images, labels_placeholder: val_labels}
                summary, loss_value, psnr = sess.run([merged, loss_op, psnr_op], feed_dict=val_feed)
                print("Step: [%2d], time: [%4.4f], validation_loss = [%.8f], validation_psnr = [%.8f]" % (step, time.time()-start_time, loss_value, psnr))
                val_writer.add_summary(summary, step)
                tr_images, tr_labels = sess.run([tr_images_op, tr_labels_op])
                tr_feed = {images_placeholder: tr_images, labels_placeholder: tr_labels}
                summary, loss_value, psnr = sess.run([merged, loss_op, psnr_op], feed_dict=tr_feed)
                print("Step: [%2d], time: [%4.4f], training_loss =  [%.8f], training_psnr = [%.8f]" % (step, time.time()-start_time, loss_value, psnr))
                train_writer.add_summary(summary, step)

            if step % FLAGS.save_step == 0 or (step+1) == FLAGS.max_steps:
                self.save(sess, FLAGS.checkpoint_dir, step=step)
        coord.request_stop()
        coord.join(threads)
        sess.close()

    def test_domain_adapted(self, FLAGS):
        with tf.Session() as sess:
            images, labels = test_setup(FLAGS)
            images_placeholder = tf.placeholder(tf.float32, [None, None, None, self.c_dim], name='images')
            labels_placeholder = tf.placeholder(tf.float32, [None, None, None, self.c_dim], name='labels')
            pred = self.inference(images_placeholder, isTrain=False)
            self.saver = tf.train.Saver(tf.global_variables())
            if self.load(sess, self.checkpoint_dir):
                print(" [*] Load SUCCESS")
            else:
                print(" [!] Load failed...")
            MSE = tf.reduce_mean(tf.square(labels_placeholder - pred))
            psnr_op = tf.multiply(tf.constant(20, dtype=tf.float32),
                                tf.log(1 / tf.sqrt(MSE)) / tf.log(tf.constant(10, dtype=tf.float32)), name='psnr')
            init_op = tf.group(tf.local_variables_initializer(), tf.global_variables_initializer())
            sess.run(init_op)
            for i in range(images.shape[0]):
                image = images[i]
                print(np.max(image))
                print(np.min(image))
                sr_image = pred.eval(feed_dict={images_placeholder: np.expand_dims(images[i], axis=0)})
                #psnr = sess.run(psnr_op, feed_dict={images_placeholder: np.expand_dims(images[i], axis=0), labels_placeholder: np.expand_dims(labels[i], axis=0)})
                #print("psnr = [%.8f]" % (psnr))
                print(np.max(sr_image))
                print(np.min(sr_image))
                sr_image = (sr_image * 255).astype(np.uint8).squeeze()
                print(sr_image)
                sr_image = shave(sr_image, (3, 3))
                image_path = os.path.join(os.getcwd(), FLAGS.sample_dir)
                image_path = os.path.join(image_path, "sr_image_{:d}.png".format(i))
                imsave(sr_image, image_path)

    def test_single_GPU(self, FLAGS):
        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            saver = tf.train.import_meta_graph('checkpoint/srcnn_21/SRCNN.model-783500.meta', clear_devices=True)
            saver.restore(sess, 'checkpoint/srcnn_21/SRCNN.model-783500')
            images, labels = test_setup(FLAGS)
            graph = tf.get_default_graph()
            pred = graph.get_tensor_by_name('gpu_0/conv3/conv3:0')
            images_placeholder = graph.get_tensor_by_name('images:0')
            #labels_placeholder = graph.get_tensor_by_name('labels:0')
            #self.get_tensors_ops_graph(sess)
            init_op = tf.group(tf.local_variables_initializer(), tf.global_variables_initializer())
            sess.run(init_op)
            for i in range(images.shape[0]):
                image = images[i]
                print(np.max(image))
                print(np.min(image))
                sr_image = pred.eval(feed_dict={images_placeholder: np.expand_dims(images[i], axis=0)})
                print(np.max(sr_image))
                print(np.min(sr_image))
                sr_image = (sr_image * 255).astype(np.uint8).squeeze()
                print(sr_image)
                sr_image = shave(sr_image, (3, 3))
                image_path = os.path.join(os.getcwd(), FLAGS.sample_dir)
                image_path = os.path.join(image_path, "sr_image_{:d}.png".format(i))
                imsave(sr_image, image_path)

    def test_multiple_CPU(self, FLAGS):
        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        videos_LR_placeholder = tf.placeholder(tf.float32, [None, None, None, None, self.c_dim], name='videos')
        videos_HR_placeholder = tf.placeholder(tf.float32, [None, None, None, None, self.c_dim], name='labels')
        pred = self.inference(
            tf.reshape(videos_LR_placeholder, [-1, 112, 112, 1]),
            isTrain=False)
        loss = self.tower_loss('cpu0', pred, tf.reshape(videos_HR_placeholder, [-1, 112, 112, 1]))
        psnr_op = tf.multiply(tf.constant(20, dtype=tf.float32),
                           tf.log(1 / tf.sqrt(loss)) / tf.log(tf.constant(10, dtype=tf.float32)), name='psnr')

        dirs = ['/hdd/wuzhenyu_sjtu/SRx3/test/part0', '/hdd/wuzhenyu_sjtu/SRx3/test/part1',
                '/hdd/wuzhenyu_sjtu/SRx3/test/part2', '/hdd/wuzhenyu_sjtu/SRx3/test/part3']
        count = 0
        test_files = []
        for dir in dirs:
            test_files += [os.path.join(dir, f) for f in
                          os.listdir(dir) if f.endswith('.tfrecords')]
        print(test_files)
        videos_LR, videos_HR, labels = input_data_video.inputs(filenames=test_files,
                                                                   batch_size=FLAGS.gpu_num * FLAGS.batch_size,
                                                                   num_epochs=1,
                                                                   num_threads=FLAGS.num_threads,
                                                                   vshape=[FLAGS.depth, FLAGS.height, FLAGS.width,
                                                                           FLAGS.nchannel],
                                                                   num_examples_per_epoch=1000,
                                                                   shuffle=False
                                                                   )
        init_op = tf.group(tf.local_variables_initializer(), tf.global_variables_initializer())
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        '''
        saver = tf.train.Saver(tf.trainable_variables())
        if os.path.isfile(FLAGS.pretrained_model):
            saver.restore(sess, FLAGS.pretrained_model)
            print('Session Restored!')
        '''
        try:
            psnr_sr_lst = []
            psnr_bc_lst = []
            videos_LR_lst = []
            videos_SR_lst = []
            labels_lst = []
            while not coord.should_stop():
                start_time = time.time()
                _videos_LR, _videos_HR, _labels = sess.run([videos_LR, videos_HR, labels])
                print("Point 0, time: [%4.4f]" % (time.time() - start_time))
                print(_labels.shape)
                _videos_LR_Y = _videos_LR[:, :, :, :, 0:1]
                _videos_LR_CbCr = _videos_LR[:, :, :, :, 1:]
                start_time = time.time()
                psnr_bc = self.compute_psnr(_videos_LR_Y, _videos_HR).eval(session=sess)
                print("Point 1, time: [%4.4f]" % (time.time() - start_time))

                start_time = time.time()
                psnr_bc_lst.append(psnr_bc)
                feed = {videos_LR_placeholder: _videos_LR_Y, videos_HR_placeholder: _videos_HR}
                videos_SR_Y, psnr_sr = sess.run([pred, psnr_op], feed_dict=feed)
                print("Point 2, time: [%4.4f]" % (time.time() - start_time))
                start_time = time.time()
                videos_LR_CbCr = _videos_LR_CbCr.reshape(_videos_LR_CbCr.shape[0] * _videos_LR_CbCr.shape[1],
                                                             _videos_LR_CbCr.shape[2], _videos_LR_CbCr.shape[3],
                                                             _videos_LR_CbCr.shape[4])
                LR_videos = _videos_LR.reshape(_videos_LR.shape[0] * _videos_LR.shape[1],
                                                   _videos_LR.shape[2], _videos_LR.shape[3],
                                                   _videos_LR.shape[4])
                print("Point 3, time: [%4.4f]" % (time.time() - start_time))
                start_time = time.time()
                LR_videos = ycbcr2rgb(LR_videos * 255)
                SR_videos = np.concatenate((videos_SR_Y, videos_LR_CbCr), axis=3)
                SR_videos = ycbcr2rgb(SR_videos * 255)
                print("Point 4, time: [%4.4f]" % (time.time() - start_time))
                start_time = time.time()
                videos_LR_lst += np.split(LR_videos, indices_or_sections=FLAGS.gpu_num * FLAGS.batch_size, axis=0)
                videos_SR_lst += np.split(SR_videos, indices_or_sections=FLAGS.gpu_num * FLAGS.batch_size, axis=0)
                labels_lst += _labels.tolist()
                psnr_sr_lst.append(psnr_sr)
                print("Point 5, time: [%4.4f]" % (time.time() - start_time))
                print("psnr_bicubic = [%.8f]" % psnr_bc)
                print("psnr_sr = [%.8f]" % psnr_sr)
        except tf.errors.OutOfRangeError:
            print('Done testing on all the examples')
        finally:
            coord.request_stop()
        coord.join(threads)
        psnr_bc_arr = np.asarray(psnr_bc_lst, dtype=np.float32)
        print("psnr_bc_mean = [%.8f]" % np.mean(psnr_bc_arr))
        print("psnr_bc_stddev = [%.8f]" % np.std(psnr_bc_arr))

        psnr_sr_arr = np.asarray(psnr_sr_lst, dtype=np.float32)
        print("psnr_sr_mean = [%.8f]" % np.mean(psnr_sr_arr))
        print("psnr_sr_stddev = [%.8f]" % np.std(psnr_sr_arr))
        # write_video(videos_SR_lst, labels_lst)
        convert_to_videos(np.asarray(videos_LR_lst), np.asarray(labels_lst),
                              'Train_LR_{}_{}'.format(FLAGS.scale, count), '/hdd/wuzhenyu_sjtu')
        convert_to_videos(np.asarray(videos_SR_lst), np.asarray(labels_lst),
                              'Train_SR_{}_{}'.format(FLAGS.scale, count), '/hdd/wuzhenyu_sjtu')
        count += 1
        sess.close()

    def test_multiple_GPU(self, FLAGS):
        conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        conf.gpu_options.allow_growth = True
        videos_LR_placeholder = tf.placeholder(tf.float32, [None, None, None, None, self.c_dim], name='videos')
        videos_HR_placeholder = tf.placeholder(tf.float32, [None, None, None, None, self.c_dim], name='labels')
        preds = []
        losses = []
        sess = tf.Session(config=conf)
        with tf.variable_scope(tf.get_variable_scope()) as scope:
            for gpu_index in range(0, FLAGS.gpu_num):
                with tf.device('/gpu:%d' % gpu_index):
                    print('/gpu:%d' % gpu_index)
                    with tf.name_scope('%s_%d' % ('gpu', gpu_index)) as scope:
                        pred = self.inference(
                            tf.reshape(videos_LR_placeholder[gpu_index * FLAGS.batch_size:
                                                    (gpu_index + 1) * FLAGS.batch_size],[-1,112,112,1]),
                            isTrain=False)
                        loss = self.tower_loss(scope,
                                               pred,
                                               tf.reshape(videos_HR_placeholder[gpu_index * FLAGS.batch_size:
                                                        (gpu_index + 1) * FLAGS.batch_size], [-1,112,112,1]))
                        losses.append(loss)
                        preds.append(pred)
                        tf.get_variable_scope().reuse_variables()
        preds = tf.concat(preds, 0)
        _loss = tf.reduce_mean(losses, name='mse')
        psnr = tf.multiply(tf.constant(20, dtype=tf.float32),
                           tf.log(1 / tf.sqrt(_loss)) / tf.log(tf.constant(10, dtype=tf.float32)), name='psnr')
        #dirs = ['/home/wuzhenyu_sjtu/train/part1']
        dirs = ['/home/wuzhenyu_sjtu/test']
        test_files = []
        for dir in dirs:
            test_files += [os.path.join(dir, f) for f in
                      os.listdir(dir) if f.endswith('.tfrecords')]
        print(test_files)
        videos_LR, videos_HR, labels = input_data_video.inputs(filenames=test_files,
                                                               batch_size=FLAGS.gpu_num*FLAGS.batch_size,
                                                               num_epochs=1,
                                                               num_threads=FLAGS.num_threads,
                                                               vshape=[FLAGS.depth, FLAGS.height, FLAGS.width,
                                                                       FLAGS.nchannel],
                                                               num_examples_per_epoch=1000,
                                                               shuffle=False
                                                               )
        init_op = tf.group(tf.local_variables_initializer(), tf.global_variables_initializer())
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        '''
        saver = tf.train.Saver(tf.trainable_variables())
        if os.path.isfile(FLAGS.pretrained_model):
            saver.restore(sess, FLAGS.pretrained_model)
            print('Session Restored!')
        '''
        count = 24
        try:
            psnr_sr_lst = []
            psnr_bc_lst = []
            videos_LR_lst = []
            videos_SR_lst = []
            labels_lst = []
            i = 0
            while not coord.should_stop():
                i += 1
                start_time = time.time()
                _videos_LR, _videos_HR, _labels = sess.run([videos_LR, videos_HR, labels])
                print("Point 0, time: [%4.4f]" % (time.time() - start_time))
                print(_labels.shape)
                _videos_LR_Y = _videos_LR[:, :, :, :, 0:1]
                _videos_LR_CbCr = _videos_LR[:, :, :, :, 1:]
                start_time = time.time()
                psnr_bc = self.compute_psnr(_videos_LR_Y, _videos_HR).eval(session=sess)
                print("Point 1, time: [%4.4f]" % (time.time() - start_time))

                start_time = time.time()
                psnr_bc_lst.append(psnr_bc)
                feed = {videos_LR_placeholder: _videos_LR_Y, videos_HR_placeholder: _videos_HR}
                videos_SR_Y, psnr_sr = sess.run([preds, psnr], feed_dict=feed)
                print("Point 2, time: [%4.4f]" % (time.time() - start_time))

                start_time = time.time()
                psnr_sr_lst.append(psnr_sr)
                '''
                videos_LR_CbCr = _videos_LR_CbCr.reshape(_videos_LR_CbCr.shape[0] * _videos_LR_CbCr.shape[1],
                                                             _videos_LR_CbCr.shape[2], _videos_LR_CbCr.shape[3],
                                                             _videos_LR_CbCr.shape[4])
                LR_videos = _videos_LR.reshape(_videos_LR.shape[0] * _videos_LR.shape[1],
                                                   _videos_LR.shape[2], _videos_LR.shape[3],
                                                   _videos_LR.shape[4])
                print("Point 3, time: [%4.4f]" % (time.time() - start_time))
                start_time = time.time()
                LR_videos = ycbcr2rgb(LR_videos * 255)
                SR_videos = np.concatenate((videos_SR_Y, videos_LR_CbCr), axis=3)
                SR_videos = ycbcr2rgb(SR_videos * 255)
                print("Point 4, time: [%4.4f]" % (time.time() - start_time))
                start_time = time.time()
                videos_LR_lst += np.split(LR_videos, indices_or_sections=FLAGS.gpu_num*FLAGS.batch_size, axis=0)
                videos_SR_lst += np.split(SR_videos, indices_or_sections=FLAGS.gpu_num*FLAGS.batch_size, axis=0)
                labels_lst += _labels.tolist()
                if i % 100 == 0:
                    convert_to_videos(np.asarray(videos_LR_lst), np.asarray(labels_lst),
                                      'Train_LR_{}_{}'.format(FLAGS.scale, count), '/hdd/wuzhenyu_sjtu')
                    convert_to_videos(np.asarray(videos_SR_lst), np.asarray(labels_lst),
                                      'Train_SR_{}_{}'.format(FLAGS.scale, count), '/hdd/wuzhenyu_sjtu')
                    count += 1
                    videos_LR_lst, videos_SR_lst, labels_lst = None, None, None
                    gc.collect()
                    videos_LR_lst, videos_SR_lst, labels_lst = [], [], []
                '''
                print("Point 5, time: [%4.4f]" % (time.time() - start_time))
                print("psnr_bicubic = [%.8f]" % psnr_bc)
                print("psnr_sr = [%.8f]" % psnr_sr)
                print('Memory usage: % 2.2f MB' % round(
                    resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1)
                      )
        except tf.errors.OutOfRangeError:
            print('Done testing on all the examples')
        finally:
            coord.request_stop()
        coord.join(threads)
        psnr_bc_arr = np.asarray(psnr_bc_lst, dtype=np.float32)
        print("psnr_bc_mean = [%.8f]" % np.mean(psnr_bc_arr))
        print("psnr_bc_stddev = [%.8f]" % np.std(psnr_bc_arr))

        psnr_sr_arr = np.asarray(psnr_sr_lst, dtype=np.float32)
        print("psnr_sr_mean = [%.8f]" % np.mean(psnr_sr_arr))
        print("psnr_sr_stddev = [%.8f]" % np.std(psnr_sr_arr))
        #write_video(videos_SR_lst, labels_lst)
        sess.close()

    def get_tensors_ops_graph(self, sess):
        tvars = tf.trainable_variables()
        tvars_vals = sess.run(tvars)
        print('----------------------------Trainable Variables-----------------------------------------')
        for var, val in zip(tvars, tvars_vals):
            print(var.name, val)
        print('----------------------------------------Operations-------------------------------------')
        for op in tf.get_default_graph().get_operations():
            print(str(op.name))
        print('----------------------------------Nodes in the Graph---------------------------------------')
        print([n.name for n in tf.get_default_graph().as_graph_def().node])

    def save(self, sess, checkpoint_dir, step):
        model_name = "SRCNN.model"
        model_dir = "%s_%s" % ("srcnn", self.label_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, sess, checkpoint_dir):
        print(" [*] Reading checkpoints...")
        model_dir = "%s_%s/models_ucfframes" % ("srcnn", self.label_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False
