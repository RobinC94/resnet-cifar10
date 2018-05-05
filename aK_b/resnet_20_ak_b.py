from __future__ import print_function
import keras
from keras.layers import Dense, Conv2D, BatchNormalization, Activation
from keras.layers import AveragePooling2D, Input, Flatten
from keras.regularizers import l2
from keras.models import Model
from aK_b.modified_conv2d import ModifiedConv2D

import numpy as np
import tensorflow as tf


#####################################################
# public API
def resnet_cifar10_builder_ak_b(input_shape=(32,32,3), num_classes=10, clusterid=None, template=None):
    template = tf.Variable(template, name='template', dtype='float32')
    cluster_index=0

    num_filters = 16
    last_num_filters = 3

    inputs = Input(shape=input_shape)
    cluster_length=last_num_filters*num_filters
    clusterid_one_layer=np.array(clusterid[cluster_index:cluster_index+cluster_length])
    clusterid_one_layer = clusterid_one_layer.reshape(num_filters, last_num_filters).T
    cluster_index += cluster_length
    last_num_filters = num_filters

    x = resnet_layer(inputs=inputs, template_tensor=template, clusterid=clusterid_one_layer)

    # Instantiate the stack of residual units
    for stack in range(3):
        for res_block in range(3):
            strides = 1
            if stack > 0 and res_block == 0:  # first layer but not first stack
                strides = 2  # downsample

            cluster_length = last_num_filters*num_filters
            clusterid_one_layer = np.array(clusterid[cluster_index:cluster_index + cluster_length])
            clusterid_one_layer = clusterid_one_layer.reshape(num_filters, last_num_filters).T
            cluster_index += cluster_length
            last_num_filters = num_filters

            y = resnet_layer(inputs=x,
                             num_filters=num_filters,
                             strides=strides,
                             template_tensor=template,
                             clusterid=clusterid_one_layer)

            cluster_length = last_num_filters * num_filters
            clusterid_one_layer = np.array(clusterid[cluster_index:cluster_index + cluster_length])
            clusterid_one_layer = clusterid_one_layer.reshape(num_filters, last_num_filters).T
            cluster_index += cluster_length
            last_num_filters = num_filters

            y = resnet_layer(inputs=y,
                             num_filters=num_filters,
                             activation=None,
                             template_tensor=template,
                             clusterid=clusterid_one_layer)

            if stack > 0 and res_block == 0:  # first layer but not first stack
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
            x = keras.layers.add([x, y])
            x = Activation('relu')(x)
        num_filters *= 2

    # Add classifier on top.
    # v1 does not use BN after last shortcut connection-ReLU
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    outputs = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal')(y)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model


#####################################################
# private API
def resnet_layer(inputs,        ## conv-bn-relu
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True,
                 template_tensor = None,
                 clusterid = None):

    if kernel_size == 3:
        conv = ModifiedConv2D(num_filters,
                              kernel_size=kernel_size,
                              strides=strides,
                              padding='same',
                              kernel_initializer='he_normal',
                              kernel_regularizer=l2(1e-4),
                              template_tensor=template_tensor,
                              clusterid=clusterid)
    else:
        conv = Conv2D(num_filters,
                      kernel_size=kernel_size,
                      strides=strides,
                      padding='same',
                      kernel_initializer='he_normal',
                      kernel_regularizer=l2(1e-4))

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x




    #
