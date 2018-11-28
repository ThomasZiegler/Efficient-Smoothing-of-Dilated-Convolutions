# allow print to be displayed in the cluster job logs
from __future__ import print_function

import argparse
import os
import tensorflow as tf
from model import Model
from utils import write_log

"""
This script defines hyperparameters.
"""



#def configure(start_step_ = 0, num_steps_ = 2000, pretrain_file_ =
#              '../reference_model/deeplab_resnet_init.ckpt', valid_step_ = 2000):
def configure():
    flags = tf.app.flags

    # training
#    flags.DEFINE_integer('start_step', start_step_, 'start number of iterations')
#    flags.DEFINE_integer('num_steps', num_steps_, 'maximum number of iterations')
    flags.DEFINE_integer('start_step', 0, 'start number of iterations')
    flags.DEFINE_integer('num_steps', 20, 'maximum number of iterations')
    flags.DEFINE_integer('save_interval', 10000, 'number of iterations for saving and visualization')
    flags.DEFINE_integer('random_seed', 1234, 'random seed')
    flags.DEFINE_float('weight_decay', 0.0005, 'weight decay rate')
    flags.DEFINE_float('learning_rate', 0.001, 'learning rate')
    flags.DEFINE_float('power', 0.9, 'hyperparameter for poly learning rate')
    flags.DEFINE_float('momentum', 0.9, 'momentum')
    flags.DEFINE_string('encoder_name', 'deeplab', 'name of pre-trained model: res101, res50 or deeplab')
#    flags.DEFINE_string('pretrain_file', pretrain_file_, 'pre-trained model filename corresponding to encoder_name')
    flags.DEFINE_string('pretrain_file', '../reference_model/deeplab_resnet_init.ckpt', 'pre-trained model filename corresponding to encoder_name')
    flags.DEFINE_string('dilated_type', 'smooth_GI', 'type of dilated conv: regular, decompose, smooth_GI, smooth_SSC or average_filter')
    flags.DEFINE_string('data_list', './dataset/train.txt', 'training data list filename')

    # train & validate
    flags.DEFINE_integer('num_iterations', 2, 'number of test & validate iterations')

    # validation
#    flags.DEFINE_integer('valid_step', valid_step_, 'checkpoint number for validation')
    flags.DEFINE_integer('valid_step', 20, 'checkpoint number for validation')
    flags.DEFINE_integer('valid_num_steps', 1449, '= number of validation samples')
    flags.DEFINE_string('valid_data_list', './dataset/val.txt', 'validation data list filename')

    # prediction / saving outputs for testing or validation
    flags.DEFINE_string('out_dir', 'output', 'directory for saving outputs')
    flags.DEFINE_integer('test_step', 50000, 'checkpoint number for testing/validation')
    flags.DEFINE_integer('test_num_steps', 1449, '= number of testing/validation samples')
    flags.DEFINE_string('test_data_list', './dataset/val.txt', 'testing/validation data list filename')
    flags.DEFINE_boolean('visual', True, 'whether to save predictions for visualization')

    # data
    flags.DEFINE_string('data_dir', '../VOC2012', 'data directory')
    flags.DEFINE_integer('batch_size', 10, 'training batch size')
    flags.DEFINE_integer('input_height', 321, 'input image height')
    flags.DEFINE_integer('input_width', 321, 'input image width')
    flags.DEFINE_integer('num_classes', 21, 'number of classes')
    flags.DEFINE_integer('ignore_label', 255, 'label pixel value that should be ignored')
    flags.DEFINE_boolean('random_scale', True, 'whether to perform random scaling data-augmentation')
    flags.DEFINE_boolean('random_mirror', True, 'whether to perform random left-right flipping data-augmentation')
    
    # log
    flags.DEFINE_string('modeldir', 'model', 'model directory')
    flags.DEFINE_string('logfile', 'log.txt', 'training log filename')
    flags.DEFINE_string('logdir', 'log', 'training log directory')

    flags.FLAGS._parse_flags()
    return flags.FLAGS

def del_all_flags(FLAGS):
    if not FLAGS.__dict__['__parsed']:
        FLAGS._parse_flags()
    for key in FLAGS.__flags.keys():
        FLAGS.__delattr__(key)


def main(_):
    parser = argparse.ArgumentParser()
    parser.add_argument('--option', dest='option', type=str, default='train',
        help='actions: train, test, predict or train_test')
    args = parser.parse_args()

    if args.option not in ['train', 'test', 'predict', 'train_test']:
        print('invalid option: ', args.option)
        print("Please input a option: train, test, predict or train_test")
    else:
        if args.option == 'train_test':
            num_iterations = 2
            num_steps = 20
            FLAGS = configure()
            FLAGS.__flags['num_steps'] = num_steps

            for i in range(0, num_iterations):
                write_log ('Iteration: %d' % (i+1), FLAGS.__flags['logfile'])


                # set flags
                start_step = i*num_steps
                valid_step = (i+1)*num_steps
                FLAGS.__flags['start_step'] = start_step
                FLAGS.__flags['valid_step'] = valid_step


                # train
                sess = tf.Session()
                model = Model(sess, FLAGS)
                getattr(model, 'train')()
                tf.reset_default_graph()

                # test
                sess = tf.Session()
                model = Model(sess, FLAGS)
                getattr(model, 'test')()
                tf.reset_default_graph()

                # load previous model
                pretrain_file = './model/model.ckpt-' + str(valid_step)
                FLAGS.__flags['pretrain_file'] = pretrain_file

                # delete model files if they exists
                try:
                    os.remove('./model/model.ckpt-' + str(start_step) + '.data-00000-of-00001')
                    os.remove('./model/model.ckpt-' + str(start_step) + '.index')
                    os.remove('./model/model.ckpt-' + str(start_step) + '.meta')
                except OSError:
                    pass

        else:
            # Set up tf session and initialize variables.
            # config = tf.ConfigProto()
            # config.gpu_options.allow_growth = True
            # sess = tf.Session(config=config)
            sess = tf.Session()
            # Run
            model = Model(sess, configure())
            getattr(model, args.option)()


if __name__ == '__main__':
    # Choose which gpu or cpu to use
    # os.environ['CUDA_VISIBLE_DEVICES'] = '7'
    tf.app.run()

