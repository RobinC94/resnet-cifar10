import os

import numpy as np
import scipy.stats as stats

import resnet
from keras.layers.convolutional import Conv2D
from termcolor import cprint

####################################
##config params
pair_layers_num = 0
r_thresh = 1
filter_size = 3
zero_thresh = 1e-4

####################################
## public API
def modify_model(model, file_save = None):

    # 1 select conv layers
    conv_layers_list = get_conv_layers_list(model)
    cprint("selected conv layers is:" + str(conv_layers_list), "red")

    # 2 get kernels stack
    kernels_stack = get_kernels_stack(model, conv_layers_list)
    print "num of searched kernels:" + str(len(kernels_stack.keys()))

    # 3 generate pairs
    pair_res = generate_pair_res(kernels_stack, file_save)
    print "num of pairs is:" + str(len(pair_res))

    # 4 modify model
    modify_kernels_stack(kernels_stack, pair_res)
    set_modified_kernels_stack_to_model(model, kernels_stack, conv_layers_list)

def load_modified_model_from_file(model, file_load = None):

    # 1 select conv layers
    conv_layers_list = get_conv_layers_list(model)
    cprint("selected conv layers is:" + str(conv_layers_list), "red")

    # 2 get kernels stack
    kernels_stack = get_kernels_stack(model, conv_layers_list)
    print "num of searched kernels:" + str(len(kernels_stack.keys()))

    # 3 get pairs
    f_load = open(file_load, "r")
    pair_res = load_pairs_from_file(f_load, kernels_stack)
    print "num of pairs is:" + str(len(pair_res))
    f_load.close()

    # 4 modify model
    modify_kernels_stack(kernels_stack, pair_res)
    set_modified_kernels_stack_to_model(model, kernels_stack, conv_layers_list)

########################################
## private API

def get_conv_layers_list(model):
    '''
        only  choose layers which is conv layer, and its filter_size must be same as param "filter_size"
    '''
    res = []
    layers = model.layers
    for i,l in enumerate(layers):
        if isinstance(l, Conv2D) and l.kernel.shape.as_list()[:2] == [filter_size, filter_size]:
            res+= [i]
    return res[:pair_layers_num]

def get_kernels_stack(model, conv_layers_list):
    kernels = []
    index = []
    kernel_num = 0

    for l in conv_layers_list:
        weights = model.layers[l].get_weights()[0]  ##0 weights, 1 bias; HWCN
        for i in range(model.layers[l].filters):  ##kernel num
            for s in range(model.layers[l].input_shape[-1]):  # kernel depth
                weights_slice = weights[:, :, s, i]  # HWCK
                if abs(weights_slice.max()) + abs(weights_slice.min()) > zero_thresh:
                    kernels += [(weights_slice)]
                    index += [(l, i, s)]
                kernel_num += 1
    print "num of total kernels:", kernel_num

    kernels_stack = {key: value for key, value in zip(index, kernels)}
    return kernels_stack


def least_square(dataa, datab):
    assert (dataa.shape == datab.shape)
    dataa = dataa.reshape(-1)
    datab = datab.reshape(-1)
    a, b = np.polyfit(dataa, datab, 1)  ##notice the direction: datab = a*dataa + b
    err = np.sum(np.abs(a * dataa + b - datab))
    return (a, b, err)

def generate_pair_res(kernels_stack, file=None):
    pair_res=[]
    id_list = kernels_stack.keys()
    kernels_list = {}

    for id in id_list:
        kernels_list[id] = [(kernels_stack[id].flatten()), 0]

    i = 0
    num_template = 0
    if file <> None:
        f = open(file, "w")

    while i < len(id_list) - 1:
        if kernels_list[id_list[i]][1] <> 0:
            i += 1
            continue

        j = i + 1
        while j < len(id_list):
            if kernels_list[id_list[j]][1] <> 0:
                j += 1
                continue
            pre = stats.pearsonr(kernels_list[id_list[i]][0], kernels_list[id_list[j]][0])[0]
            if abs(pre) > r_thresh:
                fit_res = least_square(kernels_list[id_list[i]][0], kernels_list[id_list[j]][0])
                pair_res += [(id_list[i], id_list[j], fit_res)]
                print i, j, pre, fit_res[2]
                if file <> None:
                    print >> f, i, j, fit_res[0], fit_res[1], fit_res[2]

                if kernels_list[id_list[i]][1] == 0:
                    num_template += 1
                kernels_list[id_list[i]][1] = 1
                kernels_list[id_list[j]][1] = 2
            j += 1
        i += 1

    if file <> None:
        f.close()
    print "num of template kernels:" + str(num_template)
    return pair_res

def load_pairs_from_file(f, kernels_stack):
    pair_res=[]
    id_list = kernels_stack.keys()
    temp_num = 0
    x_last = -1
    for f_line in f:
        (x, y, a, b, err)=f_line.split(' ')
        if int(x) <> x_last:
            x_last = int(x)
            temp_num += 1
        pair_res += [(id_list[int(x)], id_list[int(y)],(float(a), float(b), float(err)))]

    print "loading pairs done"
    print "num of templates is: " + str(temp_num)
    return pair_res

def modify_kernels_stack(kernels_stack, pair_res):
    id_list = kernels_stack.keys()
    for pair in pair_res:
        (k1, k2, f) = pair
        a = f[0]
        b = f[1]
        kernels_stack[k2] = a*kernels_stack[k1] + b

def set_modified_kernels_stack_to_model(model, kernels_stack, conv_layers_list):
    for l in conv_layers_list:
        weights = model.layers[l].get_weights()
        for i in range(model.layers[l].filters):  ##kernel num
            for s in range(model.layers[l].input_shape[-1]):  # kernel depth
                if kernels_stack.has_key((l, i, s)):
                    weights[0][:, :, s, i] = kernels_stack[(l, i, s)]
                else:
                    weights[0][:,:,s,i]=[[0,0,0],[0,0,0],[0,0,0]]
        model.layers[l].set_weights(weights)


#####################################
## for debug
if __name__ == "__main__":

    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    model = resnet.ResnetBuilder.build_resnet_18((3, 32, 32), 10)
    model.load_weights("./resnet/weights.h5")

    #ori_result = load_and_test(model)
    #result_names = model.metrics_names

    weights = model.layers[37].get_weights()[0]
    print weights
'''
    pair_layers_num=15
    conv_layers_list = get_conv_layers_list(model)
    cprint("selected conv layers is:" + str(conv_layers_list), "red")

    # 2 get kernels stack
    kernels_stack = get_kernels_stack(model, conv_layers_list)
    print "num of searched kernels:" + str(len(kernels_stack.keys()))

    # 3 generate pairs
    #pair_res = generate_pair_res(kernels_stack)
    #load_modified_model_from_file(model, file_load="./tmp/kernel_pairs_0.95_1.txt")

    set_modified_kernels_stack_to_model(model, kernels_stack, conv_layers_list)
'''

