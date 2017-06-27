
# coding: utf-8

# # Convolutional Neural Network applied to Transient Detection

# In[1]:

import numpy as np
import gzip
#import matplotlib.pyplot as plt
#get_ipython().magic('matplotlib inline')

# Put data loader into a function, break up potentially into train/valid/test subsets...
def load_pkl_data(chunk_num, split_frac=(0.8, 0.9), verbose=False):
    fname = 'all_chunks/chunk_%d_5000.pkl.gz' % chunk_num
    pkl_data = np.load(gzip.GzipFile(fname, 'rb'), encoding='bytes')
    if False:
        print(pkl_data.keys())
        print(pkl_data[b'diff_images'].shape)
    
    N_data = pkl_data[b'diff_images'].shape[0]
    if False:
        print(N_data)
    X = np.array([pkl_data[b'temp_images'].reshape((N_data, 21, 21)), 
                 pkl_data[b'sci_images'].reshape((N_data, 21, 21)),
                 pkl_data[b'diff_images'].reshape((N_data, 21, 21)),
                 pkl_data[b'SNR_images'].reshape((N_data, 21, 21))])
    X = np.swapaxes(X, 0, 1)

    Y = np.array([np.logical_not(pkl_data[b'labels']), pkl_data[b'labels']]).transpose()
    if False:
        print(X.shape, Y.shape)
        
    N_train = int(N_data * split_frac[0])
    N_valid = 0
    if split_frac[0] < 1.0:
        N_valid = int(N_data * split_frac[1])
    N_test = 0
    if split_frac[1] < 1.0:
        N_test  = int(N_data * 1.0)

    X_train, Y_train = X[:N_train], Y[:N_train]
    X_valid = Y_valid = None
    if N_valid > 0:
        X_valid, Y_valid = X[N_train:N_valid], Y[N_train:N_valid]
    X_test = Y_test = None
    if N_test > 0:
        X_test, Y_test = X[N_valid:N_test], Y[N_valid:N_test]

    if verbose:
        print(np.mean(Y[:,0]), np.mean(Y[:,1]))
        print("Train: ", X_train.shape, Y_train.shape)
        if N_valid > 0:
            print("Valid: ", X_valid.shape, Y_valid.shape)
        if N_test > 0:
            print("Test: ", X_test.shape, Y_test.shape)
        
    return (X_train, Y_train), (X_valid, Y_valid), (X_test, Y_test)

# We create a Keras sequential model and compile it.

# from IPython.display import SVG
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.utils.vis_utils import plot_model, model_to_dot

# set dimensions ordering (depth as index 1)
import keras
keras.backend.set_image_dim_ordering('th')
from keras.layers.advanced_activations import LeakyReLU

# 0.04 and 0.5 and 1./100000. are params from Deep-HiTS paper
def make_model(compile=True, epochs=100, lrate=0.04, dropout=0.5, decay=1./100000.,
               momentum=0.0, use_leaky=True):
    model = Sequential()
    model.add(ZeroPadding2D((3, 3), input_shape = (4, 21, 21)))
    if not use_leaky:
        model.add(Convolution2D(32, (4, 4), activation='relu'))
    else:
        model.add(Convolution2D(32, (4, 4)))
        model.add(LeakyReLU(alpha=0.01))
    model.add(ZeroPadding2D((1, 1)))
    if not use_leaky:
        model.add(Convolution2D(32, (3, 3), activation='relu'))
    else:
        model.add(Convolution2D(32, (3, 3)))
        model.add(LeakyReLU(alpha=0.01))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    if not use_leaky:
        model.add(Convolution2D(64, (3, 3), activation='relu'))
    else:
        model.add(Convolution2D(64, (3, 3)))
        model.add(LeakyReLU(alpha=0.01))
    model.add(ZeroPadding2D((1, 1)))
    if not use_leaky:
        model.add(Convolution2D(64, (3, 3), activation='relu'))
    else:
        model.add(Convolution2D(64, (3, 3)))
        model.add(LeakyReLU(alpha=0.01))
    model.add(ZeroPadding2D((1, 1)))
    if not use_leaky:
        model.add(Convolution2D(64, (3, 3), activation='relu'))
    else:
        model.add(Convolution2D(64, (3, 3)))
        model.add(LeakyReLU(alpha=0.01))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    if not use_leaky:
        model.add(Dense(64, activation='relu'))
    else:
        model.add(Dense(64))
        model.add(LeakyReLU(alpha=0.01))
    if dropout is not None:
        model.add(Dropout(dropout))
    if not use_leaky:
        model.add(Dense(64, activation='relu'))
    else:
        model.add(Dense(64))
        model.add(LeakyReLU(alpha=0.01))
    if dropout is not None:
        model.add(Dropout(dropout))
    model.add(Dense(2, activation='softmax'))
    
    if compile:
        # model.compile(loss='mean_squared_error',
        #       optimizer='sgd', metrics=['accuracy'])

        # initiate RMSprop optimizer (OLD)
        #opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

        # Compile model
        #epochs = 25
        #lrate = 0.01
        if decay is None:
            if epochs > 2:
                decay = lrate/epochs
            else:
                decay = lrate/100.
        opt = keras.optimizers.SGD(lr=lrate, momentum=momentum, decay=decay, nesterov=False)

        # Let's train the model using RMSprop
        model.compile(loss='mean_squared_error', #categorical_crossentropy',
                      optimizer=opt,
                      metrics=['accuracy'])

    return model



# Now, we fit our model to the training data-set
# This func is not used, the model is run below...

def run_model(model, train, valid, epochs=25, batch_size=32, data_augmentation=True, 
              patience=5, **kwargs):
    X_train, Y_train = train
    X_valid, Y_valid = valid
    
    from keras.callbacks import EarlyStopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=patience)

    if not data_augmentation:
        if epochs > 2:
            print('Not using data augmentation.')
        histry = model.fit(X_train, Y_train, batch_size=batch_size, 
                  epochs=epochs, validation_data=(X_valid, Y_valid),
                  shuffle=True, callbacks=[early_stopping], **kwargs)
    else:
        if epochs > 2:
            print('Using real-time data augmentation.')
        from keras.preprocessing.image import ImageDataGenerator
        # This will do preprocessing and realtime data augmentation:
        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=True)  # randomly flip images

        # Compute quantities required for feature-wise normalization
        # (std, mean, and principal components if ZCA whitening is applied).
        # datagen.fit(X_train)  # so not needed.

        # Fit the model on the batches generated by datagen.flow().
        histry = model.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size), 
                            epochs=epochs, validation_data=(X_valid, Y_valid),
                            steps_per_epoch=X_train.shape[0] // batch_size,
                            callbacks=[early_stopping], **kwargs) 
        
    return model, histry

seed = 666
np.random.seed(seed)
batch_size = 50

from keras.preprocessing.image import ImageDataGenerator

# This will do preprocessing and realtime data augmentation:
imageDatagen = ImageDataGenerator(
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
    width_shift_range=0.,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.,  # randomly shift images vertically (fraction of total height)
    horizontal_flip=True,  # randomly flip images
    vertical_flip=True)  # randomly flip images

def data_generator_train():
    data_range = np.arange(200)  # use inds 0-200 for training
    while True:
        chunk = np.random.choice(data_range, size=1)[0]
        print('TRAIN:', chunk)
        (X, Y), _, _ = load_pkl_data(chunk, split_frac=(1.0, 1.0), verbose=False)
        for i in range(X.shape[0]//batch_size):  # about 5000/32/2 = 150//2
            Xb, Yb = imageDatagen.flow(X, Y, batch_size=batch_size).next()
            yield (Xb, Yb)
        #yield (X, Y)

def data_generator_valid():
    data_range = np.arange(201, 285)  # use inds 201-284 for validation
    while True:
        chunk = np.random.choice(data_range, size=1)[0]
        print('VALID:', chunk)
        (X, Y), _, _ = load_pkl_data(chunk, split_frac=(1.0, 1.0), verbose=False)
        for i in range(X.shape[0]//batch_size): # about 5000/32/2 = 150/2
            Xb, Yb = imageDatagen.flow(X, Y, batch_size=batch_size).next()
            yield (Xb, Yb)
        #yield (X, Y)

import keras.backend as kbackend
    
class SGDLearningRateTracker(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        optimizer = self.model.optimizer
        lr = kbackend.eval(optimizer.lr * (1. / (1. + optimizer.decay * optimizer.iterations)))
        print('\nLR: %.8f\n' % lr)
        print('opt.lr=%.8f; opt.decay=%.8f; opt.iter=%.8f:'%(kbackend.eval(optimizer.lr),
                                                             kbackend.eval(optimizer.decay),
                                                             kbackend.eval(optimizer.iterations)))

# Note that setting the decay smaller (e.g. 1/100000) makes the model go to random quickly.
# Note that for the learning rate, 'iterations' is actually number of *batches* run (not number of epochs run)!
epochs = 3000
lrate = 0.04
decay = 1./10000. #1./1000.

import os.path
if os.path.isfile('best_model.hdf5'):  # link this name to your favorite model to re-load it
    print('Loading "best_model.hdf5"')
    model = keras.models.load_model('./best_model.hdf5')
    initial_epoch = 35   ## <- change this depending on restart!
else:
    model = make_model(compile=True, epochs=epochs, lrate=lrate, decay=decay)
    initial_epoch = 0

print(model.summary())

train_generator = data_generator_train()
valid_generator = data_generator_valid()

from keras.callbacks import EarlyStopping, ModelCheckpoint
early_stopping = EarlyStopping(monitor='val_loss', patience=1000)
checkpointing = ModelCheckpoint('model.{epoch:06d}-{val_loss:.6f}.hdf5', # './best_model.hdf5',
                                monitor='val_loss', verbose=1,
                                save_best_only=True, save_weights_only=False,
                                mode='auto', period=1)

rateMonitoring = SGDLearningRateTracker()

# steps_per_epoch=5000: 5000/32=156 chunks trained per epoch,
# then validation_steps=500: 500/32=15.6 chunks tested per epoch
history = model.fit_generator(generator=train_generator, 
                    validation_data=valid_generator, validation_steps=2500,
                    epochs=epochs, steps_per_epoch=2500, initial_epoch=initial_epoch,
                    callbacks=[early_stopping, checkpointing, rateMonitoring], workers=1)


import pickle, gzip
pickle.dump(gzip.GzipFile('./FINAL_trainHistoryDict.pkl.gz', 'wb'))



