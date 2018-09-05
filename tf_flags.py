import tensorflow as tf
import datetime

flags = tf.app.flags
flags.DEFINE_integer("gpu_num", 4, "Number of gpu")
flags.DEFINE_integer("max_steps", 50000, "Number of steps")
flags.DEFINE_integer("num_epochs", 50000, "Number of epochs")
#flags.DEFINE_integer("batch_size", 4096, "The size of batch images [128]")
flags.DEFINE_integer("batch_size", 32, "The size of batch images [128]")
flags.DEFINE_integer("num_examples_per_epoch", 21884, "The number of examples per epoch for training")
flags.DEFINE_integer("image_size", 33, "The size of image to use [33]")
flags.DEFINE_integer("label_size", 21, "The size of label to produce [21]")
flags.DEFINE_float("learning_rate", 1e-4, "The learning rate of gradient descent algorithm [1e-4]")
flags.DEFINE_float("momentum", 0.9, "The momentum for the momentum optimizer [0.9]")
flags.DEFINE_integer("c_dim", 1, "Dimension of image color. [1]")
flags.DEFINE_integer("scale", 8, "The size of scale factor for preprocessing input image [3]")
flags.DEFINE_integer("tr_stride", 14, "The size of stride to apply input training image [14]")
flags.DEFINE_integer("val_stride", 21, "The size of stride to apply input validation image [21]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Name of checkpoint directory [checkpoint]")
flags.DEFINE_string("sample_dir", "sample", "Name of sample directory [sample]")
flags.DEFINE_boolean("is_train", True, "True for training, False for testing [True]")
log_dir = 'tensorboard_events/{}/'.format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
flags.DEFINE_string('log_dir', log_dir, 'Directory where to write the tensorboard events')
flags.DEFINE_integer('num_threads', 10, 'Number of threads enqueuing tensor list')
flags.DEFINE_integer('val_step', 5, 'Number of steps for validation')
flags.DEFINE_integer('save_step', 50, 'Number of step to save the model')
flags.DEFINE_integer('depth', 16, 'Depth for the video')
flags.DEFINE_integer('width', 112, 'Width for the video')
flags.DEFINE_integer('height', 112, 'Height for the video')
flags.DEFINE_integer('nchannel', 1 ,'Number of channel for the video')

FLAGS = flags.FLAGS
