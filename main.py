import os

import model_modify as modi
import model_train_and_test as trte
import keras

from keras.layers.convolutional import Conv2D
from keras import initializers
from termcolor import cprint

import resnet

def print_conv_layer_info(model):
    f = open("./tmp/conv_layers_info.txt", "w")
    f.write("layer index   filter number   filter shape(HWCK)\n")

    cprint("conv layer information:", "red")
    for i, l in enumerate(model.layers):
        if isinstance(l, Conv2D):
            print i, l.filters, l.kernel.shape.as_list()
            print >> f, i, l.filters, l.kernel.shape.as_list()
    f.close()

def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    model = resnet.ResnetBuilder.build_resnet_10((3,32,32), 10)
    model.load_weights("./weights/resnet10_weights.h5")

    keras.utils.plot_model(model, to_file="./tmp/resnet10.png")
    print_conv_layer_info(model)

    ori_result = trte.load_and_test(model)

    modi.pair_layers_num = 8
    modi.r_thresh = 0.85
    trte.pair_layers_num = modi.pair_layers_num

    ## please check your file name first!!!
    ## don't cover existing pair files!!!
    file = "./tmp/resnet10_pairs_"+str(modi.r_thresh)+"_"+str(modi.pair_layers_num)+".txt"
    modi.load_modified_model_from_file(model, file_load=file)
    #modi.modify_model(model, file_save=file)

    trte.fine_tune(model)
    #trte.load_and_train(model, 200, None, True)
    test_result = trte.load_and_test(model)
    result_names = model.metrics_names

    cprint("original test result:", "blue")
    print result_names[0], ": ", ori_result[0]
    print result_names[1], ": ", ori_result[1]
    cprint("modified test result:", "blue")
    print result_names[0], ": ", test_result[0]
    print result_names[1], ": ", test_result[1]
    


if __name__ == "__main__":
    main()