import datetime
import pytz
from tensorflow.python.client import timeline
import numpy as np
import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL']="2"
#os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3,4,5"
os.environ["CUDA_VISIBLE_DEVICES"]=""
import input_data_video
from tf_flags import FLAGS
import json
import pprint

class TimeLiner:
    _timeline_dict = None

    def update_timeline(self, chrome_trace):
        # convert crome trace to python dict
        chrome_trace_dict = json.loads(chrome_trace)
        # for first run store full trace
        if self._timeline_dict is None:
            self._timeline_dict = chrome_trace_dict
        # for other - update only time consumption, not definitions
        else:
            for event in chrome_trace_dict['traceEvents']:
                # events time consumption started with 'ts' prefix
                if 'ts' in event:
                    self._timeline_dict['traceEvents'].append(event)

    def save(self, f_name):
        with open(f_name, 'w') as f:
            json.dump(self._timeline_dict, f)

pp = pprint.PrettyPrinter()
pp.pprint(FLAGS.__flags)


filenames = []
for path, subdirs, files in os.walk("/hdd/wuzhenyu_sjtu/SRx3/train/"):
    for name in files:
        filenames.append(os.path.join(path, name))
#print(filenames)

dir = "SRx3/train/part0"
#dir = "/hdd/wuzhenyu_sjtu/SRx3/train/part0"

test_files = [os.path.join(dir, f) for f in
                      os.listdir(dir) if f.endswith('.tfrecords')]
#print(test_files)


videos_LR, videos_HR, labels = input_data_video.inputs(filenames=test_files,
                                                       batch_size=FLAGS.gpu_num * FLAGS.batch_size,
                                                       num_epochs=1,
                                                       num_threads=40,
                                                       vshape=[FLAGS.depth, FLAGS.height, FLAGS.width,
                                                               FLAGS.nchannel],
                                                       num_examples_per_epoch=1000,
                                                       shuffle=False
                                                       )



#config = tf.ConfigProto(device_count = {'GPU': 0})
with tf.Session() as sess:
    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init_op, options=run_options, run_metadata=run_metadata)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)
    start_time = datetime.datetime.now(pytz.timezone('America/Chicago'))
    step = 1
    speed_lst = []
    many_runs_timeline = TimeLiner()
    try:
        while not coord.should_stop():
            _, _, _ = sess.run([videos_LR, videos_HR, labels],options=run_options, run_metadata=run_metadata)
            fetched_timeline = timeline.Timeline(run_metadata.step_stats)
            chrome_trace = fetched_timeline.generate_chrome_trace_format()
            many_runs_timeline.update_timeline(chrome_trace)
            step +=1
            end_time = datetime.datetime.now(pytz.timezone('America/Chicago'))
            sec = (end_time - start_time).total_seconds()
            speed = int((FLAGS.gpu_num * FLAGS.batch_size)/sec)
            print("[{}] time[{:6.2f}] step[{:10d}] speed[{:6d}]".format(
                    str(end_time).split(".")[0], sec, step,
                    speed
                    ))
            speed_lst.append(speed)
            start_time = end_time

    except tf.errors.OutOfRangeError:
        print("Done training after reading all data")
    finally:
        coord.request_stop()
        print("coord stopped")

    coord.join(threads)

    many_runs_timeline.save('timeline_merged_%d_steps.json' % step)

    speed_arr = np.asarray(speed_lst, dtype=np.float32)
    print("avg speed: {:6.6f}".format(np.mean(speed_arr)))
    print("stddev speed: {:6.6f}".format(np.std(speed_arr)))
    print("all done")
