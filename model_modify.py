import sys,os

import numpy as np
import scipy.stats as stats

import resnet
from keras.layers.convolutional import Conv2D
from termcolor import cprint

from Bio import Cluster

####################################
##config params
pair_layers_num = 0
kmeans_k=255
r_thresh = 1
filter_size = 3
zero_thresh = 1e-4

####################################
## public API
def modify_model(model, k=kmeans_k, file_save = None):

    # 1 select conv layers
    conv_layers_list = get_conv_layers_list(model)
    cprint("selected conv layers is:" + str(conv_layers_list), "red")

    # 2 get kernels stack
    kernels_stack = get_kernels_stack(model, conv_layers_list)
    print "num of searched kernels:" + str(len(kernels_stack.keys()))

    # 3 modify with kmeans
    modify_kernels_with_kmeans(kernels_stack, k=k, f_save=file_save)
    set_modified_kernels_stack_to_model(model, kernels_stack, conv_layers_list)

def load_modified_model_from_file(model, file_load = None):

    # 1 select conv layers
    conv_layers_list = get_conv_layers_list(model)
    cprint("selected conv layers is:" + str(conv_layers_list), "red")

    # 2 get kernels stack
    kernels_stack = get_kernels_stack(model, conv_layers_list)
    print "num of searched kernels:" + str(len(kernels_stack.keys()))

    modify_kernels_with_centroids(kernels_stack, f_load=file_load)
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
                #if abs(weights_slice.max()) + abs(weights_slice.min()) > zero_thresh:
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


def modify_kernels_with_kmeans(kernels_stack, k=kmeans_k, f_save=None):
    kernels_keys=kernels_stack.keys()
    kernels_num=len(kernels_keys)

    kernels_array=np.zeros((kernels_num, filter_size**2))
    for i in range(kernels_num):
        kernel_id=kernels_keys[i]
        kernels_array[i]=kernels_stack[kernel_id].flatten()

    print "start clustering"

    clusterid, error, nfound = Cluster.kcluster(kernels_array, nclusters=k,dist='a')

    cdata, cmask = Cluster.clustercentroids(kernels_array,clusterid=clusterid,)

    print "end clustering"

    avg_sum=0
    for i in range(kernels_num):
        kernel_id = kernels_keys[i]
        cent_id=clusterid[i]
        kernel=kernels_array[i]
        cent=cdata[cent_id]
        a,b,err=least_square(cent,kernel)
        r=abs(stats.pearsonr(kernel,cent)[0])
        avg_sum+=r
        kernels_stack[kernel_id]=a*cent.reshape((3,3))+b
    avg=avg_sum/kernels_num

    print "average r2:",avg

    if f_save != None:
        f_clusterid=f_save+"_clusterid.npy"
        f_cdata=f_save+"_cdata.npy"
        np.save(f_clusterid, clusterid)
        np.save(f_cdata,cdata)

    return kernels_stack

def modify_kernels_with_centroids(kernels_stack,f_load=None):
    kernels_keys = kernels_stack.keys()
    kernels_num = len(kernels_keys)

    kernels_array = np.zeros((kernels_num, filter_size ** 2))
    for i in range(kernels_num):
        kernel_id = kernels_keys[i]
        kernels_array[i] = kernels_stack[kernel_id].flatten()

    try:
        f_clusterid=f_load+"_clusterid.npy"
        f_cdata = f_load + "_cdata.npy"
        clusterid=np.load(f_clusterid)
        cdata=np.load(f_cdata)
    except:
        print "cannot load file!"
        sys.exit(0)

    print "load centroid data done"

    avg_sum = 0
    for i in range(kernels_num):
        kernel_id = kernels_keys[i]
        cent_id = clusterid[i]
        kernel = kernels_array[i]
        cent = cdata[cent_id]
        a, b, err = least_square(cent, kernel)
        r = abs(stats.pearsonr(kernel, cent)[0])
        avg_sum += r
        kernels_stack[kernel_id] = a * cent.reshape((3, 3)) + b
    avg = avg_sum / kernels_num

    print "average r2:", avg

    return kernels_stack

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

