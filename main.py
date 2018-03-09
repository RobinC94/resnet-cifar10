import os
import keras

from keras.layers.convolutional import Conv2D
from termcolor import cprint
from keras.utils import multi_gpu_model

from resnet20 import resnet_cifar10_builder

from model_modify import cluster_model_kernels, modify_model
from model_train_and_test import model_train, model_test
from modified_conv2d import ModifiedConv2D


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
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    model = resnet_cifar10_builder(n = 3, version=1, input_shape=(32,32,3))
    #model = multi_gpu_model(model, 2)
    model.load_weights("./weights/resnet20_cifar10_weights.183.h5")

    keras.utils.plot_model(model, to_file="./tmp/resnet_v1.png")
    print_conv_layer_info(model)

    kmeans_k = 4096

    cluster_id, temp_kernels = cluster_model_kernels(model, k=kmeans_k, t=5)

    #model_new = modify_model(model, cluster_id, temp_kernels)
    modify_model(model, cluster_id, temp_kernels)
    #model_train(model, 32, 200,)
    #print_conv_layer_info(model_new, modified=True)
    model_test(model)


if __name__ == "__main__":
    main()