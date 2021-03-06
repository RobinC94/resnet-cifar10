import os

import numpy as np

from keras.layers.convolutional import Conv2D
from keras.layers.normalization import BatchNormalization
from keras.layers import Dense
from termcolor import cprint
from array import array
from Bio import Cluster

from resnet20_ak import resnet_cifar10_builder_ak

####################################
##config params
kmeans_k=256
filter_size = 3
d_thresh = 0

####################################
## public API
def cluster_model_kernels_ak(model, k=kmeans_k, t = 1):
    # 1 get 3x3 conv layers
    conv_layers_list = get_conv_layers_list(model)
    cprint("selected conv layers is:" + str(conv_layers_list), "red")

    # 2 get kernels array
    kernels_array = get_kernels_array(model, conv_layers_list)
    print "num of kernels:" + str(np.shape(kernels_array)[0])

    # 3 get clusterid and temp
    cluster_id, temp_kernels = cluster_kernels(kernels_array, k=k, times=t)

    return cluster_id, temp_kernels

def modify_model_ak(model, cluster_id, temp_kernels):
    # 1 get 3x3 conv layers
    conv_layers_list = get_conv_layers_list(model)
    cprint("selected conv layers is:" + str(conv_layers_list), "red")

    # 2 get kernels array
    kernels_array = get_kernels_array(model, conv_layers_list)
    print "num of kernels:" + str(np.shape(kernels_array)[0])

    # 3 get coefficient a
    coef_a = get_coefficient_a(kernels_array, cluster_id, temp_kernels)

    # 4 build modified model
    model_new = resnet_cifar10_builder_ak(n = 3, input_shape=(32,32,3))
    print 'rebuilding model done'

    # 5 set new model weights
    set_cluster_weights_to_model(model,model_new,
                                 clusterid=cluster_id,
                                 cdata=temp_kernels,
                                 coef_a=coef_a,
                                 conv_layers_list=conv_layers_list)
    set_cluster_weights_to_old_model(model, cluster_id,temp_kernels,coef_a,conv_layers_list)

    return model_new

def save_cluster_result_ak(clusterid, temp, f):
    f_clusterid = f + "_ak_clusterid.npy"
    f_temp = f + "_ak_temp.npy"
    np.save(f_clusterid, clusterid)
    np.save(f_temp, temp)

def load_cluster_result_ak(f):
    f_clusterid = f + "_ak_clusterid.npy"
    f_temp = f + "_ak_temp.npy"
    clusterid = np.load(f_clusterid)
    temp = np.load(f_temp)

    print 'loading cluster result done.'
    return clusterid, temp

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
    return res

def get_weighted_layers_list(model):
    '''
        get all layers with weights without conv3x3
    '''
    res = []
    layers = model.layers
    for i,l in enumerate(layers):
        if (isinstance(l, Conv2D) and l.kernel.shape.as_list()[:2] != [filter_size, filter_size]) \
                or isinstance(l, BatchNormalization) or isinstance(l, Dense):
            res += [i]
    return res

def get_kernels_array(model, conv_layers_list):
    kernels_num = 0

    kernels_buf=array('d')
    for l in conv_layers_list:
        weights = model.layers[l].get_weights()[0]  ##0 weights, 1 bias; HWCN
        for i in range(model.layers[l].filters):  ##kernel num
            for s in range(model.layers[l].input_shape[-1]):  # kernel depth
                weights_slice = weights[:, :, s, i]  # HWCK
                for w in weights_slice.flatten():
                    kernels_buf.append(w)
                kernels_num+=1

    kernels_array = np.frombuffer(kernels_buf,dtype=np.float).reshape(kernels_num,filter_size**2)
    return kernels_array

def cluster_kernels(kernels_array, k=kmeans_k, times=1):
    print "start clustering"

    clusterid = []
    error_best = float('inf')
    for i in range(times):
        clusterid_single, error, nfound = Cluster.kcluster(kernels_array, nclusters=k, dist='x')
        if error < error_best:
            clusterid = clusterid_single
            error_best = error
    print 'error:', error_best

    cdata, cmask = Cluster.clustercentroids(kernels_array, clusterid=clusterid, )

    print "end clustering"

    return clusterid, cdata

def propotional_fitting(kernel, temp):
    assert (kernel.shape == temp.shape)
    kernel=kernel.reshape(-1)
    temp = temp.reshape(-1)
    # k = a * t
    # a = sum(xiyi)/sum(xi^2)
    a_num = np.sum(kernel * temp)
    a_denom = np.sum(temp**2)
    #assert(abs(a_denom) < 1e-33)
    a = a_num/a_denom
    err = abs(kernel - a * temp)

    return a, err

def get_coefficient_a(kernels_array, clusterid, cdata):
    kernels_num = np.shape(kernels_array)[0]
    coef_a=[]

    avg=0
    for i in range(kernels_num):
        cent_id = clusterid[i]
        kernel = kernels_array[i]
        cent = cdata[cent_id]
        a, err = propotional_fitting(kernel, cent)
        coef_a += [a]
        avg += get_cos_dis(kernel,cent)/kernels_num


    print "average cos: %6.4f\t"%(avg)

    return coef_a

def get_cos_dis(a, b):
    num = sum(a*b)
    denom=np.linalg.norm(a) * np.linalg.norm(b)
    #assert(abs(denom) < 1e-32)
    return 1 - abs(num / denom)

def set_cluster_weights_to_model(model, model_new, clusterid, cdata, coef_a, conv_layers_list):
    kernel_id = 0
    for l in conv_layers_list:
        weights = model.layers[l].get_weights()
        weights_new = model_new.layers[l].get_weights()
        weights_new[1] = weights[1]
        for i in range(model.layers[l].filters):  ##kernel num
            for s in range(model.layers[l].input_shape[-1]):  # kernel depth
                weights_new[0][s, i] = coef_a[kernel_id]

                cent_id = clusterid[kernel_id]
                cent = cdata[cent_id]
                weights_new[2][:, :, s, i] = np.array(cent).reshape(filter_size, filter_size)  # HWCK
                kernel_id += 1

        model_new.layers[l].set_weights(weights_new)

    weighted_layers_list = get_weighted_layers_list(model)
    for l in weighted_layers_list:
        weights = model.layers[l].get_weights()
        model_new.layers[l].set_weights(weights)

def set_cluster_weights_to_old_model(model, clusterid, cdata, coef_a, conv_layers_list):
    kernels_id = 0
    for l in conv_layers_list:
        weights = model.layers[l].get_weights()
        for i in range(model.layers[l].filters):  ##kernel num
            for s in range(model.layers[l].input_shape[-1]):  # kernel depth
                cent = clusterid[kernels_id]
                temp = cdata[cent]
                a = coef_a[kernels_id]
                weights[0][:, :, s, i] = np.array(a * temp).reshape(filter_size, filter_size)
                kernels_id += 1
        model.layers[l].set_weights(weights)


#####################################
## for debug
if __name__ == "__main__":

    os.environ['CUDA_VISIBLE_DEVICES'] = ''