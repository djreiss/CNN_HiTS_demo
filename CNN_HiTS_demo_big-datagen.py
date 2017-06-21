
# coding: utf-8

# # Convolutional Neural Network applied to Transient Detection

# In[1]:

import numpy as np
import gzip
#import matplotlib.pyplot as plt
#get_ipython().magic('matplotlib inline')


# Put data loader into a function, break up potentially into train/valid/test subsets...

# In[2]:

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


# In[3]:

(X_train, Y_train), (X_test, Y_test), _ = load_pkl_data(188, (0.9, 1.0), verbose=True)


# We create a Keras sequential model and compile it.

# In[4]:

from IPython.display import SVG
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.utils.vis_utils import plot_model, model_to_dot

# set dimensions ordering (depth as index 1)
import keras
keras.backend.set_image_dim_ordering('th')

def make_model(compile=True, epochs=100, lrate=0.01, decay=None):
    model = Sequential()
    model.add(ZeroPadding2D((3, 3), input_shape = (4, 21, 21)))
    model.add(Convolution2D(32, (4, 4), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    if epochs <= 2:
        model.add(Dropout(0.1))
    model.add(Dense(64, activation='relu'))
    if epochs <= 2:
        model.add(Dropout(0.1))
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
        opt = keras.optimizers.SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)

        # Let's train the model using RMSprop
        model.compile(loss='mean_squared_error', #categorical_crossentropy',
                      optimizer=opt,
                      metrics=['accuracy'])

    return model


# In[5]:

epochs = 25
model = make_model(epochs=25)


# In[6]:

print(model.summary())


# Now, we fit our model to the training data-set

# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[107]:

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
model = make_model()

# model, histry = run_model(model, (X_train, Y_train), (X_valid, Y_valid), data_augmentation=False)score = model.evaluate(X_test, Y_test)
# print('Test loss:', score[0])
# print('Test accuracy:', score[1])np.random.seed(seed)
# model = make_model()
# model, histry = run_model(model, (X_train, Y_train), (X_valid, Y_valid), data_augmentation=True,
#                          patience=10)
# score = model.evaluate(X_test, Y_test)
# print('Test loss:', score[0])
# print('Test accuracy:', score[1])from sklearn import metrics
# pred = model.predict_classes(X_test)
# print(metrics.classification_report(Y_test[:,1].astype(int), pred))

#fpr, tpr, thresholds = metrics.roc_curve(Y_test[:,1].astype(int), pred)
#plt.plot(fpr, tpr, label='ROC Curve')
#plt.xlabel('FPR'); plt.ylabel('TPR (recall)')ypred = pred
ytest = Y_test[:, 1].astype(int)

# N_plot = 10
# only_plot_wrong = True
# if not only_plot_wrong:
#     plot_inds = range(N_plot)
# else:
#     plot_inds = np.where(ypred != ytest)[0]
#     if len(plot_inds) > N_plot:
#         plot_inds = plot_inds[:N_plot]
# N_plot = len(plot_inds)

# plt.clf()
# fig, axes = plt.subplots(N_plot, 4, figsize=(4, N_plot*1.2),
#                         subplot_kw={'xticks': [], 'yticks': []})
# i = 0
# for ind in plot_inds:
#     axes.flat[4*i].imshow(X_test[ind][0], interpolation = "none")
#     axes.flat[4*i + 1].imshow(X_test[ind][1], interpolation = "none")
#     axes.flat[4*i + 2].imshow(X_test[ind][2], interpolation = "none")
#     axes.flat[4*i + 3].imshow(X_test[ind][3], interpolation = "none")

#     axes.flat[4*i + 3].set_title ("predicted pbb = " + str(np.round(ypred[ind], 2)) + 
#                                   ", label = " + str(ytest[ind]))
#     i += 1
# plt.show()
# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# Now try fitting a large number of data in batches:

# In[7]:

n_data = 100  # number of datasets to run
seed = 666
batch_size = 32
epochs = 1  # 10   # Probably want to stop around 25 but now we have auto-stopping


# In[94]:

# np.random.seed(seed)
# model = make_model()

# for iter in range(100:)
#     for d in range(n_data):
#         (X_train, Y_train), (X_valid, Y_valid), (X_test, Y_test) = load_pkl_data(d)
#         _, history = run_model(model, (X_train, Y_train), (X_valid, Y_valid), 
#                   data_augmentation=True, verbose=0)
#         score = model.evaluate(X_test, Y_test, verbose=0)
#         print('Iter %d, Dataset %d: Test loss = %f; Test accuracy = %f' % (iter, d, score[0], score[1]))
#         if iter > 0 and iter % 10 == 0:
#             model.save('models/CNN_HiTS_demo_big_%02d.hdf5' % iter)


# # In[95]:

# model.save('models/CNN_HiTS_demo_big_FINAL.hdf5')


# In[ ]:

# fpr, tpr, thresholds = metrics.roc_curve(Y_test[:,1].astype(int), pred)
# plt.plot(fpr, tpr, label='ROC Curve')
# plt.xlabel('FPR'); plt.ylabel('TPR (recall)')


# In[ ]:




# In[ ]:




# In[ ]:




# Try using `fit_generator()` with a data generator.

# In[8]:

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



# In[22]:

batch_size = 32

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

# In[ ]:

epochs = 1000
model = make_model(compile=True, epochs=epochs, lrate=0.001, decay=1/400.)

train_generator = data_generator_train()
valid_generator = data_generator_valid()

from keras.callbacks import EarlyStopping, ModelCheckpoint
early_stopping = EarlyStopping(monitor='val_loss', patience=3000)
checkpointing = ModelCheckpoint('./best_model.hdf5', monitor='val_loss', verbose=1,
                                save_best_only=True, save_weights_only=False,
                                mode='auto', period=1)

rateMonitoring = SGDLearningRateTracker()

model.fit_generator(generator=train_generator, 
                    validation_data=valid_generator, validation_steps=1,
                    epochs=epochs, steps_per_epoch=400,
                    callbacks=[early_stopping, checkpointing, rateMonitoring], workers=1)


# In[ ]:



