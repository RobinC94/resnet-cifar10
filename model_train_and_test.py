import os

import numpy as np
import keras.backend as K

from keras.optimizers import Adam
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.layers.convolutional import Conv2D
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping, ModelCheckpoint
from keras.datasets import cifar10
from termcolor import cprint

####################################
##config params
pair_layers_num = 0
filter_size = 3

####################################
## public API
def fine_tune(model, epoch = 20, data_path=None, data_argumentation=True):

    # fix paired layers
    conv_layers_list = get_conv_layers_list(model)
    cprint("fixed conv layers is:" + str(conv_layers_list), "red")
    for layer_index in conv_layers_list:
        model.layers[layer_index].trainable = False
    # fine tune
    load_and_train(model, epoch, data_path, data_argumentation)
    cprint("fine tune is done\n", "yellow")

def load_and_test(model, data_path=None):
    batch_size = 32
    nb_classes = 10

    (X_train, y_train), (X_test, y_test) = load_data(data_path)

    Y_test = np_utils.to_categorical(y_test, nb_classes)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    X_test -= mean_image
    X_train /= 128.
    X_test /= 128.

    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=1e-4), metrics=['accuracy'])
    result = model.evaluate(x=X_test, y=Y_test, batch_size=batch_size, verbose=1)

    print '\n'
    return result

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
            res += [i]
    return res[:pair_layers_num]

def load_data(data_path=None):
    if data_path == None:
        path = './cifar-10-batches-py/'
    else:
        path = data_path

    num_train_samples = 50000

    x_train = np.zeros((num_train_samples, 3, 32, 32), dtype='uint8')
    y_train = np.zeros((num_train_samples,), dtype='uint8')

    for i in range(1, 6):
        fpath = os.path.join(path, 'data_batch_' + str(i))
        data, labels = cifar10.load_batch(fpath)
        x_train[(i - 1) * 10000: i * 10000, :, :, :] = data
        y_train[(i - 1) * 10000: i * 10000] = labels

    fpath = os.path.join(path, 'test_batch')
    x_test, y_test = cifar10.load_batch(fpath)

    y_train = np.reshape(y_train, (len(y_train), 1))
    y_test = np.reshape(y_test, (len(y_test), 1))

    if K.image_data_format() == 'channels_last':
        x_train = x_train.transpose(0, 2, 3, 1)
        x_test = x_test.transpose(0, 2, 3, 1)

    print "loading data done."

    return (x_train, y_train), (x_test, y_test)

def load_and_train(model, epoch = 200, data_path=None, data_augmentation=True):
    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)
    early_stopper = EarlyStopping(min_delta=0.001, patience=10)
    csv_logger = CSVLogger('fine_tune_resnet18_cifar10.csv')
    ckpt = ModelCheckpoint(filepath="./fine_tune_weights.h5", monitor='loss', save_best_only=True,
                           save_weights_only=True)

    batch_size = 32
    nb_classes = 10
    nb_epoch = epoch
    data_augmentation = True

    (X_train, y_train), (X_test, y_test) = load_data(data_path)

    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    # subtract mean and normalize
    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    X_test -= mean_image
    X_train /= 128.
    X_test /= 128.

    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=1e-4), metrics=['accuracy'])

    if not data_augmentation:
        print('Not using data augmentation.')
        model.fit(X_train, Y_train,
                  batch_size=batch_size,
                  nb_epoch=nb_epoch,
                  validation_data=(X_test, Y_test),
                  shuffle=True,
                  callbacks=[lr_reducer, early_stopper, csv_logger, ckpt])
    else:
        print('Using real-time data augmentation.')
        # This will do preprocessing and realtime data augmentation:
        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False)  # randomly flip images

        # Compute quantities required for featurewise normalization
        # (std, mean, and principal components if ZCA whitening is applied).
        datagen.fit(X_train)

        # Fit the model on the batches generated by datagen.flow().
        model.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size),
                            steps_per_epoch=X_train.shape[0] // batch_size,
                            validation_data=(X_test, Y_test),
                            epochs=nb_epoch, verbose=1, max_q_size=100,
                            callbacks=[lr_reducer, early_stopper, csv_logger, ckpt],
                            workers = 8,
                            use_multiprocessing = True)


#####################################
## for debug
if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = ''