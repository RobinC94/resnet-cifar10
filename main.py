import os

from keras.layers.convolutional import Conv2D
from termcolor import cprint

from resnet20 import resnet_cifar10_builder
from aK_b.model_modify import modify_model, cluster_model_kernels, \
                              save_cluster_result, load_cluster_result
from model_train_and_test import fine_tune, model_test, model_train
from aK_b.modified_conv2d import ModifiedConv2D

from aK.model_modify_ak import modify_model_ak, cluster_model_kernels_ak, \
                               save_cluster_result_ak, load_cluster_result_ak

import numpy as np


def print_conv_layer_info(model, modified = False):
    f = open("./tmp/conv_layers_info.txt", "w")
    f.write("layer index   filter number   filter shape(HWCK)\n")

    cprint("conv layer information:", "red")

    layer_type = Conv2D
    if modified:
        layer_type = ModifiedConv2D

    for i, l in enumerate(model.layers):
        if isinstance(l, layer_type):
            print i, l.filters, l.kernel.shape.as_list()
            print >> f, i, l.filters, l.kernel.shape.as_list()
    f.close()

def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    model = resnet_cifar10_builder(n = 3, input_shape=(32,32,3))
    #model = multi_gpu_model(model, 2)
    model.load_weights("./weights/resnet20_cifar10_weights.183.h5")

    #keras.utils.plot_model(model, to_file="./tmp/resnet_v1.png")
    print_conv_layer_info(model)

    kmeans_k = 2048
    file = "./tmp/resnet20_test_" + str(kmeans_k)

    cluster_id, temp_kernels = cluster_model_kernels_ak(model, k=kmeans_k, t=1)
    save_cluster_result_ak(cluster_id, temp_kernels, file)
    #cluster_id, temp_kernels = load_cluster_result_ak(file)

    model_new = modify_model_ak(model, cluster_id, temp_kernels)

    #model_new = resnet_cifar10_builder(n=3, input_shape=(32, 32, 3), modified=True)
    #model_new.load_weights('./weights/resnet20_cifar10_fine_tune_weights.256.h5')

    #model_train(model, 32, 200,)
    #print_conv_layer_info(model_new, modified=True)
    model_test(model_new)
    #model_test(model)
    #fine_tune(model_new)

if __name__ == "__main__":
    main()