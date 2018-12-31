import tensorflow as tf
import numpy as np
import six
import sys



"""
This script provides different 2d dilated convolutions.

I appreciate ideas for a more efficient implementation of the proposed two smoothed dilated convolutions.
"""



def _dilated_conv2d(dilated_type, x, kernel_size, num_o, dilation_factor, name,
                    top_scope, biased=False):
    if dilated_type == 'regular':
        return _regular_dilated_conv2d(x, kernel_size, num_o, dilation_factor, name, top_scope, biased)
    elif dilated_type == 'decompose':
        return _decomposed_dilated_conv2d(x, kernel_size, num_o, dilation_factor, name, top_scope, biased)
    elif dilated_type == 'smooth_GI':
        return _smoothed_dilated_conv2d_GI(x, kernel_size, num_o, dilation_factor, name, top_scope, biased)
    elif dilated_type == 'smooth_SSC':
        return _smoothed_dilated_conv2d_SSC(x, kernel_size, num_o, dilation_factor, name, top_scope, biased)
    elif dilated_type == 'average_filter':
        return _averaged_dilated_conv2d(x, kernel_size, num_o, dilation_factor, name, top_scope, biased)
    elif dilated_type == 'gaussian_filter':
        return _gaussian_dilated_conv2d_oneLearned(x, kernel_size, num_o, dilation_factor, name, top_scope, biased)
    elif dilated_type == 'aggregation':
        return _combinational_layer(x, kernel_size, num_o, dilation_factor, name, top_scope, biased)



    else:
        print('dilated_type ERROR!')
        print("Please input: regular, decompose, smooth_GI or smooth_SSC")
        sys.exit(-1)

def _regular_dilated_conv2d(x, kernel_size, num_o, dilation_factor, name,
                            top_scope, biased=False):
    """
    Dilated conv2d without BN or relu.
    """
    num_x = x.shape[3].value
    with tf.variable_scope(name) as scope:
        w = tf.get_variable('weights', shape=[kernel_size, kernel_size, num_x, num_o])
        o = tf.nn.atrous_conv2d(x, w, dilation_factor, padding='SAME')
        if biased:
            b = tf.get_variable('biases', shape=[num_o])
            o = tf.nn.bias_add(o, b)
        return o

def _averaged_dilated_conv2d(x, kernel_size, num_o, dilation_factor, name, top_scope, biased=False):
    """
    Dilated conv2d with antecedent average filter and without BN or relu.
    """
    num_x = x.shape[3].value

    filter_size = dilation_factor - 1
    # perform averaging (as seprable convolution)
    w_avg_value = 1.0/(filter_size*filter_size)
    w_avg = tf.Variable(tf.constant(w_avg_value,
                                    shape=[filter_size,filter_size,num_x,1]), name='w_avg')
    o = tf.nn.depthwise_conv2d_native(x, w_avg, [1,1,1,1], padding='SAME')

    with tf.variable_scope(name) as scope:
        w = tf.get_variable('weights', shape=[kernel_size, kernel_size, num_x, num_o])
        o = tf.nn.atrous_conv2d(o, w, dilation_factor, padding='SAME')
        if biased:
            b = tf.get_variable('biases', shape=[num_o])
            o = tf.nn.bias_add(o, b)
        return o



def _gaussian_dilated_conv2d_fix(x, kernel_size, num_o, dilation_factor, name, top_scope, biased=False):
    """
    Dilated conv2d with antecedent gaussian filter and without BN or relu.
    """
    num_x = x.shape[3].value
    filter_size = dilation_factor - 1

    # perform gaussian filtering (as seprable convolution)
    sigma = 1.00

    # create kernel grid
    ax = np.arange(-filter_size // 2 + 1., filter_size // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2)) 


    mask = np.zeros([filter_size,filter_size, 1, 1, 1], dtype=np.float32)
    mask[:, :, 0, 0, 0] = kernel

    w_gauss_value = tf.Variable(tf.constant(0.0,
                                shape=[filter_size,filter_size, 1,1,1]), name='w_gauss_value',trainable=False)

    # create gaussian filter 
    w_gauss_value = tf.add(w_gauss_value, tf.constant(mask, dtype=tf.float32))
    w_gauss_value = tf.div(w_gauss_value, tf.exp(2.0 * sigma**2))
    w_gauss_value = tf.div(w_gauss_value, tf.reduce_sum(w_gauss_value))

    # perform separable convolution
    o_gauss = tf.expand_dims(x, -1)
    o_gauss = tf.nn.conv3d(o_gauss, w_gauss_value, strides=[1,1,1,1,1], padding='SAME')
    o_gauss = tf.squeeze(o_gauss, -1)

    with tf.variable_scope(name) as scope:
        # perform dilated convolution 
        w = tf.get_variable('weights', shape=[kernel_size, kernel_size, num_x, num_o])
        o = tf.nn.atrous_conv2d(o_gauss, w, dilation_factor, padding='SAME')
        if biased:
            b = tf.get_variable('biases', shape=[num_o])
            o = tf.nn.bias_add(o, b)
        return o





def _gaussian_dilated_conv2d_oneLearned(x, kernel_size, num_o, dilation_factor, name, top_scope, biased=False):
    """
    Dilated conv2d with antecedent gaussian filter and without BN or relu.
    """
    num_x = x.shape[3].value
    filter_size = dilation_factor - 1

    sigma = _get_sigma(top_scope)

    # create kernel grid
    ax = np.arange(-filter_size // 2 + 1., filter_size // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2)) 

    mask = np.zeros([filter_size,filter_size, 1, 1, 1], dtype=np.float32)
    mask[:, :, 0, 0, 0] = kernel

    w_gauss_value = tf.Variable(tf.constant(0.0,
                                shape=[filter_size,filter_size, 1,1,1]), name='w_gauss_value',trainable=False)

    # create gaussian filter 
    w_gauss_value = tf.add(w_gauss_value, tf.constant(mask, dtype=tf.float32))
    w_gauss_value = tf.div(w_gauss_value, tf.exp(2.0 * sigma**2))
    w_gauss_value = tf.div(w_gauss_value, tf.reduce_sum(w_gauss_value))

    # perform separable convolution
    o_gauss = tf.expand_dims(x, -1)
    o_gauss = tf.nn.conv3d(o_gauss, w_gauss_value, strides=[1,1,1,1,1], padding='SAME')
    o_gauss = tf.squeeze(o_gauss, -1)

    with tf.variable_scope(name) as scope:
        # perform dilated convolution 
        w = tf.get_variable('weights', shape=[kernel_size, kernel_size, num_x, num_o])
        o = tf.nn.atrous_conv2d(o_gauss, w, dilation_factor, padding='SAME')
        if biased:
            b = tf.get_variable('biases', shape=[num_o])
            o = tf.nn.bias_add(o, b)
        return o


def _get_sigma(name):
    """
    Return sigma vector 
    """
    with tf.variable_scope(name) as scope:
        # init sigma if not already done
        try:
            sigma = tf.get_variable('gauss_sigma', shape=[1], initializer=tf.constant_initializer([1.0]))

        # get sigma if already initialized
        except ValueError:
            scope.reuse_variables()
            sigma = tf.get_variable('gauss_sigma')

    return sigma



def _gaussian_dilated_conv2d_allLearned(x, kernel_size, num_o, dilation_factor, name, top_scope, biased=False):
    """
    Dilated conv2d with antecedent gaussian filter and without BN or relu.
    """
    num_x = x.shape[3].value
    filter_size = dilation_factor - 1

    with tf.variable_scope(name) as scope:
        # perform gaussian filtering (as seprable convolution)
        # init sigma value with 1
        sigma_init = 1.00
        init = tf.constant_initializer(sigma_init)
        sigma = tf.get_variable('gauss_sigma', shape=[1], initializer=init)

        # create kernel grid
        ax = np.arange(-filter_size // 2 + 1., filter_size // 2 + 1.)
        xx, yy = np.meshgrid(ax, ax)
        kernel = np.exp(-(xx**2 + yy**2)) 

        mask = np.zeros([filter_size,filter_size, 1, 1, 1], dtype=np.float32)
        mask[:, :, 0, 0, 0] = kernel

        w_gauss_value = tf.Variable(tf.constant(0.0,
                                    shape=[filter_size,filter_size, 1,1,1]), name='w_gauss_value',trainable=False)

        # create gaussian filter 
        w_gauss_value = tf.add(w_gauss_value, tf.constant(mask, dtype=tf.float32))
        w_gauss_value = tf.div(w_gauss_value, tf.exp(2.0 * sigma**2))
        w_gauss_value = tf.div(w_gauss_value, tf.reduce_sum(w_gauss_value))

        # perform separable convolution
        o_gauss = tf.expand_dims(x, -1)
        o_gauss = tf.nn.conv3d(o_gauss, w_gauss_value, strides=[1,1,1,1,1], padding='SAME')
        o_gauss = tf.squeeze(o_gauss, -1)

        # perform dilated convolution 
        w = tf.get_variable('weights', shape=[kernel_size, kernel_size, num_x, num_o])
        o = tf.nn.atrous_conv2d(o_gauss, w, dilation_factor, padding='SAME')
        if biased:
            b = tf.get_variable('biases', shape=[num_o])
            o = tf.nn.bias_add(o, b)
        return o



def _combinational_layer(x, kernel_size, num_o, dilation_factor, name, top_scope, biased=False):
    """
    Combination of Gaussian, Average, SSC prefilter together with non per-filtered input
    """
    num_x = x.shape[3].value
    fix_w_size = dilation_factor * 2 - 1
    filter_size = dilation_factor - 1

    # perform average filtering (as seprable convolution)
    w_avg_value = 1.0/(filter_size*filter_size)
    w_avg = tf.Variable(tf.constant(w_avg_value,
                                    shape=[filter_size,filter_size,num_x,1]), name='w_avg')
    o_avg = tf.nn.depthwise_conv2d_native(x, w_avg, [1,1,1,1], padding='SAME')

    # perform gaussian filtering 
    sigma = 1.0
    ax = np.arange(-filter_size // 2 + 1., filter_size // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)

    kernel = np.exp(-(xx**2 + yy**2)) 
    w_gauss_value = tf.Variable(tf.constant(0.0,
                                shape=[filter_size,filter_size, 1,1,1]), name='w_gauss_value',trainable=False)

    mask = np.zeros([filter_size,filter_size, 1, 1, 1], dtype=np.float32)
    mask[:, :, 0, 0, 0] = kernel

    # create gaussian filter 
    w_gauss_value = tf.add(w_gauss_value, tf.constant(mask, dtype=tf.float32))
    w_gauss_value = tf.div(w_gauss_value, tf.exp(2.0 * sigma**2))
    w_gauss_value = tf.div(w_gauss_value, tf.reduce_sum(w_gauss_value))

    # perform separable convolution
    o_gauss = tf.expand_dims(x, -1)
    o_gauss = tf.nn.conv3d(o_gauss, w_gauss_value, strides=[1,1,1,1,1], padding='SAME')
    o_gauss = tf.squeeze(o_gauss, -1)

    # get c vector
    c_ = _get_c_vector(top_scope)

    with tf.variable_scope(name) as scope:
        # perform SSC convolution
        fix_w = tf.get_variable('fix_w', shape=[fix_w_size, fix_w_size, 1, 1, 1], initializer=tf.zeros_initializer)
        mask = np.zeros([fix_w_size, fix_w_size, 1, 1, 1], dtype=np.float32)
        mask[dilation_factor - 1, dilation_factor - 1, 0, 0, 0] = 1
        fix_w = tf.add(fix_w, tf.constant(mask, dtype=tf.float32))
        o_ssc = tf.expand_dims(x, -1)
        o_ssc = tf.nn.conv3d(o_ssc, fix_w, strides=[1,1,1,1,1], padding='SAME')
        o_ssc = tf.squeeze(o_ssc, -1)

        # perform aggregation (combine pre filters)
        o = c_[0]*x + c_[1]*o_avg + c_[2]*o_gauss + c_[3]*o_ssc

        # perform dilated convolution
        w = tf.get_variable('weights', shape=[kernel_size, kernel_size, num_x, num_o])
        o = tf.nn.atrous_conv2d(o, w, dilation_factor, padding='SAME')
        if biased:
            b = tf.get_variable('biases', shape=[num_o])
            o = tf.nn.bias_add(o, b)
        return o


def _get_c_vector(name):
    """
    Return c vector 
    """
    with tf.variable_scope(name) as scope:
        # init vector if not already done
        try:
            c_ = tf.get_variable('c_vector', shape=[4], initializer=tf.constant_initializer([0.25, 0.25, 0.25, 0.25]))

        # get vector if already initialized
        except ValueError:
            scope.reuse_variables()
            c_ = tf.get_variable('c_vector')

    # perform soft-max to ensure values in [0,1] 
    c_.assign(tf.nn.softmax(c_))
    return c_



def _decomposed_dilated_conv2d(x, kernel_size, num_o, dilation_factor, name, top_scope, biased=False):
    """
    Decomposed dilated conv2d without BN or relu.
    """
    # padding so that the input dims are multiples of dilation_factor
    H = tf.shape(x)[1]
    W = tf.shape(x)[2]
    pad_bottom = (dilation_factor - H % dilation_factor) if H % dilation_factor != 0 else 0
    pad_right = (dilation_factor - W % dilation_factor) if W % dilation_factor != 0 else 0
    pad = [[0, pad_bottom], [0, pad_right]]
    # decomposition to smaller-sized feature maps
    # [N,H,W,C] -> [N*d*d, H/d, W/d, C]
    o = tf.space_to_batch(x, paddings=pad, block_size=dilation_factor)
    # perform regular conv2d
    num_x = x.shape[3].value
    with tf.variable_scope(name) as scope:
        w = tf.get_variable('weights', shape=[kernel_size, kernel_size, num_x, num_o])
        s = [1, 1, 1, 1]
        o = tf.nn.conv2d(o, w, s, padding='SAME')
        if biased:
            b = tf.get_variable('biases', shape=[num_o])
            o = tf.nn.bias_add(o, b)
    o = tf.batch_to_space(o, crops=pad, block_size=dilation_factor)
    return o

def _smoothed_dilated_conv2d_GI(x, kernel_size, num_o, dilation_factor, name, top_scope, biased=False):
    """
    Smoothed dilated conv2d via the Group Interaction (GI) layer without BN or relu.
    """
    # padding so that the input dims are multiples of dilation_factor
    H = tf.shape(x)[1]
    W = tf.shape(x)[2]
    pad_bottom = (dilation_factor - H % dilation_factor) if H % dilation_factor != 0 else 0
    pad_right = (dilation_factor - W % dilation_factor) if W % dilation_factor != 0 else 0
    pad = [[0, pad_bottom], [0, pad_right]]
    # decomposition to smaller-sized feature maps
    # [N,H,W,C] -> [N*d*d, H/d, W/d, C]
    o = tf.space_to_batch(x, paddings=pad, block_size=dilation_factor)
    # perform regular conv2d
    num_x = x.shape[3].value
    with tf.variable_scope(name) as scope:
        w = tf.get_variable('weights', shape=[kernel_size, kernel_size, num_x, num_o])
        s = [1, 1, 1, 1]
        o = tf.nn.conv2d(o, w, s, padding='SAME')
        fix_w = tf.Variable(tf.eye(dilation_factor*dilation_factor), name='fix_w')
        l = tf.split(o, dilation_factor*dilation_factor, axis=0)
        os = []
        for i in six.moves.range(0, dilation_factor*dilation_factor):
            os.append(fix_w[0, i] * l[i])
            for j in six.moves.range(1, dilation_factor*dilation_factor):
                os[i] += fix_w[j, i] * l[j]
        o = tf.concat(os, axis=0)
        if biased:
            b = tf.get_variable('biases', shape=[num_o])
            o = tf.nn.bias_add(o, b)
    o = tf.batch_to_space(o, crops=pad, block_size=dilation_factor)
    return o

def _smoothed_dilated_conv2d_SSC(x, kernel_size, num_o, dilation_factor, name, top_scope, biased=False):
    """
    Smoothed dilated conv2d via the Separable and Shared Convolution (SSC) without BN or relu.
    """
    num_x = x.shape[3].value
    fix_w_size = dilation_factor * 2 - 1
    with tf.variable_scope(name) as scope:
        fix_w = tf.get_variable('fix_w', shape=[fix_w_size, fix_w_size, 1, 1, 1], initializer=tf.zeros_initializer)
        mask = np.zeros([fix_w_size, fix_w_size, 1, 1, 1], dtype=np.float32)
        mask[dilation_factor - 1, dilation_factor - 1, 0, 0, 0] = 1
        fix_w = tf.add(fix_w, tf.constant(mask, dtype=tf.float32))
        o = tf.expand_dims(x, -1)
        o = tf.nn.conv3d(o, fix_w, strides=[1,1,1,1,1], padding='SAME')
        o = tf.squeeze(o, -1)
        w = tf.get_variable('weights', shape=[kernel_size, kernel_size, num_x, num_o])
        o = tf.nn.atrous_conv2d(o, w, dilation_factor, padding='SAME')
        if biased:
            b = tf.get_variable('biases', shape=[num_o])
            o = tf.nn.bias_add(o, b)
        return o
