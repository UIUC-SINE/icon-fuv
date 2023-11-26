import os, glob
import numpy as np
import datetime as dt

#from cnn import icon_io as io
import tensorflow as tf

from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Activation, Add, Subtract, PReLU, Layer, ReLU
from keras.constraints import maxnorm
from keras.layers.convolutional import Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling2D
from keras.optimizers import Adadelta, Adam, SGD, RMSprop#, schedules
from keras.losses import mean_squared_error, mean_absolute_error
from keras import regularizers, Input, backend
from keras.callbacks import ModelCheckpoint, EarlyStopping

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from matplotlib import pyplot as plt

# import artifactremoval as ar

class LogLayer(Layer):
    def call(self, inputs):
        inputs_log = tf.math.log(inputs)
        return inputs_log

class ExpLayer(Layer):
    def call(self, inputs):
        inputs_exp = tf.math.exp(inputs)
        return inputs_exp

def cnn_star_detection(im_shape=(256,6,1), kernel_size=(4,2),
                      num_outputs=3,
                      nfilters=32,
                      normalization_on=True,
                      dropout_on=True,
                      optimizer = 'Adam',):

    # Create the model
    model = Sequential()

    model.add(Conv2D(nfilters, kernel_size, input_shape=im_shape, padding='same'))
    if normalization_on:  model.add(BatchNormalization())
    model.add(Activation('relu'))
    if dropout_on:  model.add(Dropout(0.1))


    model.add(Conv2D(nfilters, (4, 2), padding='same'))
    if normalization_on:
        model.add(BatchNormalization())
    model.add(Activation('relu'))

    if dropout_on:
        model.add(Dropout(0.1))

    #From (256, 6) => (64,3)
    model.add(MaxPooling2D(pool_size=(4, 2)))



    model.add(Conv2D(nfilters*2, (4, 2), padding='same'))
    if normalization_on:
        model.add(BatchNormalization())
    model.add(Activation('relu'))# , activity_regularizer=regularizers.l2(l2=0.01))

    if dropout_on:
        model.add(Dropout(0.1))

    #From (64, 3) => (16,1)
    model.add(MaxPooling2D(pool_size=(4, 3))) #,strides=(2,2)


    model.add(Conv2D(nfilters*4, (4, 1), padding='same'))
    if normalization_on:
        model.add(BatchNormalization())

    model.add(Activation('relu'))# , activity_regularizer=regularizers.l2(l2=0.01))
    if dropout_on:
        model.add(Dropout(0.1))


    #From (16, 1) => (4,1)
    model.add(MaxPooling2D(pool_size=(4, 1))) #,strides=(2,2)

    model.add(Conv2D(nfilters*8, (4, 1), padding='same'))
    if normalization_on:
        model.add(BatchNormalization())
    model.add(Activation('relu'))# , activity_regularizer=regularizers.l2(l2=0.01))
    if dropout_on:
        model.add(Dropout(0.1))


    #From (4, 1) => (1,1)
    model.add(MaxPooling2D(pool_size=(4, 1))) #,strides=(2,2)

    #From (1,1) => nfilters*8
    model.add(Flatten())
    if dropout_on:
        model.add(Dropout(0.1))

    model.add(Dense(nfilters*8, kernel_constraint=maxnorm(3)))
    if normalization_on:
        model.add(BatchNormalization())
    model.add(Activation('relu'))# , activity_regularizer=regularizers.l2(l2=0.01))
    if dropout_on:
        model.add(Dropout(0.1))


    model.add(Dense(nfilters*4, kernel_constraint=maxnorm(3)))
    if normalization_on:
        model.add(BatchNormalization())

    model.add(Activation('relu'))# , activity_regularizer=regularizers.l2(l2=0.01))
    if dropout_on:
        model.add(Dropout(0.1))


    model.add(Dense(num_outputs))
    model.add(Activation('linear'))
#     model.add(Activation('softmax'))

    loss = MeanSquaredError()

    if optimizer.lower() == 'adam':
        opt = Adam(lr=1e-3)
    else:
        opt = SGD(lr=1e-2, momentum=0.9, decay=1e-2 / 5e1)

    model.compile(loss=loss, optimizer=opt, metrics=["accuracy"])

    print(model.summary())

    return(model)

def create_tpu_model(**kwargs):

    # detect and init the TPU
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)

    # instantiate a distribution strategy
    tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)


    with tpu_strategy.scope():
        mod = denoise_net(**kwargs)

    return(mod)

def save_model(mod, name='./model'):

    save_locally = tf.saved_model.SaveOptions(experimental_io_device='/job:localhost')
    mod.save(name, options=save_locally)

    return

def restore_model(path='../input/cnnstarremoval10l64f/denoise_net_15_64_4',
                 learning_rate=1e-4,
                 total_variation_weight=0):

    if total_variation_weight>0:
        custom_loss = wrapper_loss(total_variation_weight)

        mod = load_model(path, custom_objects={'custom_loss': custom_loss})
    else:
        mod = load_model(path)

#     lr = backend.eval(mod.optimizer.lr)
#     print('Past learning rate: ', lr)

#     backend.set_value(mod.optimizer.lr, learning_rate)
#     lr = backend.get_value(mod.optimizer.lr)
#     print('Current learning rate: ', lr)

    return(mod)

def _create_denoise_layers(input,
                          kernel_size=(11,1),
                          noise_kernel_size=(1,1),
                          nfilters=4,
                          nlayers=10,
                          normalization_on=False,
                          dropout_on=False,
                          prefix='',
                          ):

    x = input
#     x_noisy = Conv2D(1, noise_kernel_size, padding='same')(x)
#     x_noisy = PReLU()(x_noisy)
#     x_noisy = ReLU()(x_noisy)

    for l in range(nlayers):
        x = Conv2D(nfilters, kernel_size, padding='same',
                   name=prefix + 'conv2D_inner_%02d' %l,
#                    kernel_regularizer=regularizers.l1(regularizer_alpha),
#                    bias_regularizer=regularizers.l2(1e-4)
                  )(x)

        if normalization_on: x = BatchNormalization()(x)

#         x = PReLU()(x)
        x = ReLU()(x)

        if dropout_on: x = Dropout(0.1)(x)

        x_pattern = Conv2D(1, noise_kernel_size, padding='same',
                           name=prefix + 'conv2D_outer_%02d' %l,
#                       kernel_regularizer=regularizers.l1(regularizer_alpha),
#                       bias_regularizer=regularizers.l2(1e-4)
                      )(x)
#         x_tmp = PReLU()(x_pattern)
        x_tmp = ReLU()(x_pattern)

        if l == 0:
            x_out = x_pattern
        else:
            x_out = Add(name=prefix + 'output_add_%02d' %l)([x_out, x_pattern])

    return(x_out)

def denoise_net(im_shape=(256,6,1),
                  kernel_size=(11,1),
                  noise_kernel_size=(1,1),
                  nfilters=128,
                  nlayers=10,
                  normalization_on=False,
                  dropout_on=False,
                  depth_multiplier=64,
                  optimizer = 'Adam',
                  learning_rate = 1e-4,
                  decay_steps = 1e3,
                  decay_rate = 0,
                  regularizer_alpha=1e-4,
                  loss='mse',
                  total_variation_weight=0,
                  offset = -50.,
               ):

    # Create the model
    input_img = Input(shape=im_shape)

    #Log scale to remove the very bright stars
    input_offset = input_img - tf.constant(offset)
    input_img_log = LogLayer()(input_offset)

    x_stars_log = _create_denoise_layers(input_img_log,
                                      kernel_size=kernel_size,
                                      noise_kernel_size=noise_kernel_size,
                                      nfilters=nfilters,
                                      nlayers=nlayers,
                                      normalization_on=normalization_on,
                                      dropout_on=dropout_on,
                                      prefix='log_',
                                      ) #Learn in log scale

    x_log = Subtract(name='log_output')([input_img_log, x_stars_log])

    #Remove the strongest undesired signal
    x = ExpLayer()(x_log)

    x_stars = _create_denoise_layers(x,
                                  kernel_size=kernel_size,
                                  noise_kernel_size=noise_kernel_size,
                                  nfilters=nfilters,
                                  nlayers=nlayers,
                                  normalization_on=normalization_on,
                                  dropout_on=dropout_on,
                                  prefix='linear_',
                                  ) #Learn (residual) in linear scale

    #Remove residual from signal without peaks
    x = Subtract(name='linear_output')([x, x_stars])
    x = Add(name='after_linear')([x, offset*tf.ones_like(x)])
    # x = x + tf.constant(offset)

    model = Model(input_img, x)

    lr_schedule = schedules.ExponentialDecay(
                    initial_learning_rate=learning_rate,
                    decay_steps=decay_steps,
                    decay_rate=0.9)

    if optimizer.lower() == 'sgd':
        opt = SGD(learning_rate=learning_rate, momentum=0.9, decay=decay_rate)
    elif optimizer.lower() == 'amsgrad':
        opt = Adam(learning_rate=lr_schedule, amsgrad=True)
    elif optimizer.lower() == 'adadelta+schedule':
        opt = Adadelta(learning_rate=lr_schedule)
    elif optimizer.lower() == 'adam+schedule':
        opt = Adam(learning_rate=lr_schedule)
    elif optimizer.lower() == 'rmsprop':
        opt = RMSprop(learning_rate=learning_rate, rho=0.9, epsilon=1e-08, decay=0.0)
    elif optimizer.lower() == 'adam':
        opt = Adam(learning_rate=learning_rate) #, beta_1=0.99, epsilon=1e-1)
    else:
        raise('No optimizer selected')

#     loss = mean_squared_error(input_img, x)

#     if total_variation_weight > 0:
#         loss_tv = total_variation_weight*tf.reduce_sum(tf.image.total_variation(x))
#         loss += loss_tv

#     # Add loss to model
#     model.add_loss(loss)

    if total_variation_weight > 0:
        loss = wrapper_loss(total_variation_weight)

    model.compile(loss=loss, optimizer=opt, metrics=["mse"])
    model.summary()

    return(model)

def autoencoder_net(im_shape=(256,6,1),
                    nfilters=256,
                    nlayers=4,
                    kernel_size=(3,1),
                    noise_kernel_size=(3,1),
                    optimizer = 'Adam',
                    learning_rate = 1e-4,
                    regularizer_alpha=1e-4,
                    loss='mse',
                    total_variation_weight=0,
               ):

    # Create the model
    input_img = Input(shape=im_shape)

    x0 = Conv2D(nfilters, kernel_size, padding='same',
               activation='relu',
               )(input_img)
#     x0 = PReLU()(x0)
    x0 = MaxPooling2D((2, 1), padding='same')(x0)
    #Output (128,6)

    x1 = Conv2D(nfilters, kernel_size, padding='same',
               activation='relu',
               )(x0)
#     x1 = PReLU()(x1)
    x1 = MaxPooling2D((2, 1), padding='same')(x1)
    #Output (64,6)

    x2 = Conv2D(nfilters, kernel_size, padding='same',
               activation='relu',
               )(x1)
#     x2 = PReLU()(x2)
    x2 = MaxPooling2D((2, 1), padding='same')(x2)
    #Output (32,6)

    x3 = Conv2D(nfilters, kernel_size, padding='same',
               activation='relu',
               )(x2)
#     x3 = PReLU()(x3)
    x3 = MaxPooling2D((2, 1), padding='same')(x3)
    #Output (16,6)

    x4 = Conv2D(nfilters, kernel_size, padding='same',
               activation='relu',
               )(x3)
#     x4 = PReLU()(x4)
    x4 = MaxPooling2D((2, 1), padding='same')(x4)
    #Output (8,6)

    x5 = Conv2D(nfilters, kernel_size, padding='same',
               activation='relu',
               )(x4)
#     x5 = PReLU()(x5)
    x5 = MaxPooling2D((2, 2), padding='same')(x5)
    #Output (4,3)

    #Input (256,6)
    xo = Conv2D(nfilters, noise_kernel_size, padding='same',
               activation='relu',
               )(input_img)
    #Input (256,6)

    #Input (128,6)
#     xo_0 = Conv2D(1, noise_kernel_size, padding='same')(x0)
#     xo_0 = PReLU()(xo_0)
#     xo_0 = UpSampling2D((2, 1))(xo_0)
    xo_0 = Conv2DTranspose(1, noise_kernel_size, padding='same',
                           strides=(2,1),
                           activation='relu',
                          )(x0)
#     xo_0 = PReLU()(xo_0)
    #Output (256,6)

    #Input (64,6)
#     xo_1 = Conv2D(1, noise_kernel_size, padding='same')(x1)
#     xo_1 = PReLU()(xo_1)
#     xo_1 = UpSampling2D((4, 1))(xo_1)
    xo_1 = Conv2DTranspose(1, noise_kernel_size, padding='same',
                           strides=(4,1),
                           activation='relu',
                          )(x1)
#     xo_1 = PReLU()(xo_1)
    #Output (256,6)

    #Input (32,6)
#     xo_2 = Conv2D(1, noise_kernel_size, padding='same')(x2)
#     xo_2 = PReLU()(xo_2)
#     xo_2 = UpSampling2D((8, 1))(xo_2)
    xo_2 = Conv2DTranspose(1, noise_kernel_size, padding='same',
                           strides=(8,1),
                           activation='relu',
                          )(x2)
#     xo_2 = PReLU()(xo_2)
    #Output (256,6)

    #Input (16,6)
#     xo_3 = Conv2D(1, noise_kernel_size, padding='same')(x3)
#     xo_3 = PReLU()(xo_3)
#     xo_3 = UpSampling2D((16, 1))(xo_3)
    xo_3 = Conv2DTranspose(1, noise_kernel_size, padding='same',
                           strides=(16,1),
                           activation='relu',
                          )(x3)
#     xo_3 = PReLU()(xo_3)
    #Output (256,6)

    #Input (8,6)
#     xo_4 = Conv2D(1, noise_kernel_size, padding='same')(x4)
#     xo_4 = PReLU()(xo_4)
#     xo_4 = UpSampling2D((32, 1))(xo_4)
    xo_4 = Conv2DTranspose(1, noise_kernel_size, padding='same',
                           strides=(32,1),
                           activation='relu',
                          )(x4)
#     xo_4 = PReLU()(xo_4)
    #Output (256,6)

    #Input (4,3)
#     xo_5 = Conv2D(1, noise_kernel_size, padding='same')(x5)
#     xo_5 = PReLU()(xo_5)
#     xo_5 = UpSampling2D((64, 2))(xo_5)
    xo_5 = Conv2DTranspose(1, noise_kernel_size, padding='same',
                           strides=(64,2),
                           activation='relu',
                          )(x5)
#     xo_5 = PReLU()(xo_5)
    #Output (256,6)

    x_undesired = Add()([xo, xo_0, xo_1, xo_2, xo_3, xo_4, xo_5])

    x = Subtract()([input_img, x_undesired])

    model = Model(input_img, x)

    if optimizer.lower() == 'sgd':
        opt = SGD(learning_rate=learning_rate, momentum=0.9, decay=decay_rate)
    elif optimizer.lower() == 'rmsprop':
        opt = RMSprop(learning_rate=learning_rate, rho=0.9, epsilon=1e-08, decay=0.0)
    elif optimizer.lower() == 'adam':
        opt = Adam(learning_rate=learning_rate, beta_1=0.99, epsilon=1e-1)
    else:
        raise('No optimizer selected')

    if total_variation_weight > 0:
        loss = wrapper_loss(total_variation_weight)

    model.compile(loss=loss, optimizer=opt, metrics=["mse"])
    model.summary()

    return(model)

def total_variation(images):

    ndims = images.get_shape().ndims

    if ndims == 3:
        # The input is a single image with shape [height, width, channels].

        # Calculate the difference of neighboring pixel-values.
        # The images are shifted one pixel along the height and width by slicing.
        pixel_dif1 = tf.pow(images[1:, :, :] - images[:-1, :, :], 2)
        pixel_dif2 = tf.pow(images[:, 1:, :] - images[:, :-1, :], 2)

        # Sum for all axis. (None is an alias for all axis.)
        sum_axis = None
    elif ndims == 4:
        # The input is a batch of images with shape:
        # [batch, height, width, channels].

        # Calculate the difference of neighboring pixel-values.
        # The images are shifted one pixel along the height and width by slicing.
        pixel_dif1 = tf.pow(images[:, 1:, :, :] - images[:, :-1, :, :], 2)
        pixel_dif2 = tf.pow(images[:, :, 1:, :] - images[:, :, :-1, :], 2)

        # Only sum for the last 3 axis.
        # This results in a 1-D tensor with the total variation for each image.
        sum_axis = [1, 2, 3]
    else:
        raise ValueError('\'images\' must be either 3 or 4-dimensional.')

    # Calculate the total variation by taking the absolute value of the
    # pixel-differences and summing over the appropriate axis.
    tot_var = (
                tf.reduce_sum(pixel_dif1, axis=sum_axis) +
                tf.reduce_sum(pixel_dif2, axis=sum_axis)
                )

    tot_var = tf.reduce_sum(tot_var)

    return(tot_var)

def wrapper_loss(alpha=1e-1):

    def custom_loss(y_true, y_pred):

        loss_mse = mean_squared_error(y_pred, y_true)
        if alpha > 0:
            loss_tv = alpha*total_variation(y_pred)
            loss = loss_mse + alpha*loss_tv #+ 1e-6*tf.reduce_sum(y_pred)
        else:
            loss = loss_mse

        return(loss)

    return(custom_loss)

def vgg_net(im_shape=(256,6,1),
                      kernel_size=(3,1),
                      nfilters=64,
                      optimizer = 'Adam',
                      learning_rate = 1e-3,
                      nlayers=5,
                      normalization_on=True,
                      dropout_on=True):

    # Create the model
    input_img = Input(shape=im_shape)

    x = input_img

    x_noisy = Conv2D(1, kernel_size, padding='same')(x)

    for l in range(nlayers-1):
        x = Conv2D(nfilters, kernel_size, padding='same')(x)
        if normalization_on: x = BatchNormalization()(x)
        x = Activation("relu")(x)
        if dropout_on: x = Dropout(0.2)(x)

        x_tmp = Conv2D(1, (3,2), padding='same')(x)
        x_noisy = Add()([x_noisy, x_tmp])

    x = Subtract()([input_img, x_noisy])

    model = Model(input_img, x)

    loss = MeanSquaredError()

    if optimizer.lower() == 'adam':
        opt = Adam(lr=learning_rate)
    else:
        opt = SGD(lr=learning_rate, momentum=0.9, decay=1e-2 / 5e1)

    model.compile(loss=loss, optimizer=opt, metrics=["mse", "mae", "mape"])

    model.summary()

    return(model)

def _padding(image, nx_out=256):

    nx, ny = image.shape

    if nx == nx_out:
        im_out = np.where( np.isnan(image), 0, image)
        return(im_out)

    if nx > nx_out:
        raise ImplementedError('Padding function cannot handle input image is larger than output image')

    im_out = np.zeros( (nx_out, ny), dtype=image.dtype )

    im_out[nx_out-nx:,:] = image
    im_out[:nx_out-nx,:] = np.nanmean(image[:nx_out-nx], axis=1)[None,:]

    im_out = np.where( np.isnan(im_out), 0, im_out)

    return(im_out)

def load_data(rpath = '/Users/miguel/Data/UIUC/Instruments/ICON/L1_synthetic',
             output='image',
             scaler=None,
             ena_scaling=False,
             version=1.1,
             stars_key=None,
             clean_key=None):

    if version == 1.0:
        if clean_key is None: clean_key = 'image_ori'
        if stars_key is None: stars_key = 'image_stars'

    if version == 1.1:
        if clean_key is None: clean_key = 'image_clean'
        if stars_key is None: stars_key = 'image_stars'



    if version == 1.0:
        file_list = glob.glob( os.path.join(rpath, '*/*39.npy') )
    if version == 1.1:
        file_list = glob.glob( os.path.join(rpath, '*/*00.npy') )


    if len(file_list) < 1:
        raise

    file_list = sorted(file_list)

    X = []
    y = []

    for this_file in file_list:

        d = np.load(this_file, allow_pickle=True).item()

        im_stars = d[stars_key]
        im_clean = d[clean_key]

        im_stars = _padding(im_stars, nx_out=256)
        im_clean = _padding(im_clean, nx_out=256)

        x0 = np.ravel(d['x0'])[0]
        y0 = np.ravel(d['y0'])[0]
        amp = np.ravel(d['amp'])[0]

        # update our corresponding batches lists
        X.append( im_stars )

        if output == 'image':
            y.append( im_clean )
        else:
            y.append( np.array([x0, y0, amp]) )

    nt, nx, ny = np.array(X).shape

#     fuv_mode = np.ones(nt)*2

#     X = ar.artifact_removal( np.transpose( X, (2,1,0) ), channel=1, fuv_mode=fuv_mode )
#     X = np.transpose( X, (2,1,0) )

    if ena_scaling:

        X = np.reshape(X, (nt,-1))
        y = np.reshape(y, (nt,-1))

        scaler = StandardScaler()
        scaler.fit(X)

        X = scaler.transform(X)

    X = np.reshape(X, (nt, nx, ny, 1))

    if output == 'image':
        if ena_scaling: y = scaler.transform(y)
        y = np.reshape(y, (nt, nx, ny, 1))
    else:
        y = np.reshape(y, (nt, 3))

    return(X, y, scaler)


def get_scaler(rpath='/Users/miguel/Data/UIUC/Instruments/ICON/L1_synthetic'):

    files = glob.glob( os.path.join(rpath, '*/*.npy') )

    #Use 100 images to calculate the scalers
    np.random.seed(171)

    i = 0
    X = []
    while True:
        d = np.load(files[i], allow_pickle=True).item()

        i += np.random.randint(1,30,size=1)[0]
        #Load 100 files and calculate the scalers
        if len(X) > 500:
            break

        im_stars = d['image_stars']
        im_stars = np.where( np.isnan(im_stars), 1e-20, im_stars)

        X.append(im_stars)

    X = np.reshape(X, (len(X), -1))

#     scaler = MinMaxScaler()
    scaler = StandardScaler()
    scaler.fit(X)

    return(scaler)

def data_generator(rpath = '/Users/miguel/Data/UIUC/Instruments/ICON/L1_synthetic',
                   bs=32,
                   mode="train",
                   scaler=None,
                   ena_scaling=True,
                   output='image',
                   version=4.0,
                   offset=-50,
                  ):

    #To keep consistency
    np.random.seed(171)

    clean_key = 'image_ori'
    if version == 1.1:
        clean_key = 'image_clean'

    # list the npy files for reading
    if mode=='train':
        if version == 1.0:
            files = glob.glob( os.path.join(rpath, '*/*[0-2]?.npy') )
        if version == 1.1:
            files = glob.glob( os.path.join(rpath, '*/*0?.npy') )
        if version >= 4.0:
            files = glob.glob( os.path.join(rpath, '*/*[0-2]?.npy') )
    elif mode=='validate':
        if version == 1.0:
            files = glob.glob( os.path.join(rpath, '*/*3[0-4].npy') )
        if version == 1.1:
            files = glob.glob( os.path.join(rpath, '*/*10.npy') )
        if version >= 4.0:
            files = glob.glob( os.path.join(rpath, '*/*30.npy') )
    else:
        if version == 1.0:
            files = glob.glob( os.path.join(rpath, '*/*3[5-9].npy') )
        if version == 1.1:
            files = glob.glob( os.path.join(rpath, '*/*10.npy') )
        if version >= 4.0:
            files = glob.glob( os.path.join(rpath, '*/*30.npy') )

    nfiles = len(files)
#     files = sorted(files)

    i = 0
    # loop indefinitely
    while True:
        # initialize our batches of images and labels
        X = []
        y = []

        # keep looping until we reach our batch size
        while len(X) < bs:
            # attempt to read the next file
            d = np.load(files[i], allow_pickle=True).item()

            i += 1 #np.random.randint(40, size=1)[0]
            if i >= nfiles:
#                 print('\nAll files read, restarting index to 0\n')
                i = 0

            im_stars = d['image_stars']
            im_clean = d[clean_key]
#             im_clean = d['image_smooth_201']

            im_stars = _padding(im_stars, nx_out=256)
            im_clean = _padding(im_clean, nx_out=256)

            im_stars = np.where(im_stars > offset, im_stars, offset)
            im_clean = np.where(im_clean > offset, im_clean, offset)

            x0 = np.ravel(d['x0'])[0]
            y0 = np.ravel(d['y0'])[0]
            amp = np.ravel(d['amp'])[0]

            # update our corresponding batches lists
            X.append( im_stars )

            if output == 'image':
                y.append( im_clean )
            else:
                y.append( np.array([x0, y0, amp]) )

            ##################
            #Adding diversity
            #################

#             if np.random.randn(1)[0] < 0:
#                 ##########################################
#                 #Add diversity flipping the image left/right
#                 im_stars2 = np.fliplr(im_stars)
#                 im_clean2 = np.fliplr(im_clean)

#                 y0 = im_clean.shape[1] - y0 - 1

#             else:
#                 ##########################################
#                 #Add diversity flipping the image up/down
#                 im_stars2 = np.flipud(im_stars)
#                 im_clean2 = np.flipud(im_clean)

#                 x0 = im_clean.shape[0] - x0 - 1

#             # update our corresponding batches lists
#             X.append( im_stars2 )

#             if output == 'image':
#                 y.append( im_clean2 )
#             else:
#                 y.append( np.array([x0, y0, amp]) )

        nt, nx, ny = np.array(X).shape

        if ena_scaling:

            X = np.reshape(X, (nt,-1))
            y = np.reshape(y, (nt,-1))

            scaler = StandardScaler()
            scaler.fit(X)

            X = scaler.transform(X)

        X = np.reshape(X, (nt, nx, ny, 1))

        if output == 'image':
            if ena_scaling: y = scaler.transform(y)
            y = np.reshape(y, (nt, nx, ny, 1))
        else:
            y = np.reshape(y, (nt, 3))

        # yield the batch to the calling function
        yield (X, y)

def train(model, X, y, num_outputs=3, epochs=20):

    nt, nx, ny = X.shape
    #Scale inputs and outputs
    X = np.reshape(X, (nt, -1))
    y = np.reshape(y, (nt, num_outputs))

    scaler_x = StandardScaler()
    scaler_y = StandardScaler()

#     scaler_x = MinMaxScaler()
#     scaler_y = MinMaxScaler()

    print(scaler_x.fit(X))
    xscale = scaler_x.transform(X)

    print(scaler_y.fit(y))
    yscale = scaler_y.transform(y)
    xscale = np.reshape(xscale, (-1, nx, ny, 1))

    X_train, X_test, y_train, y_test = train_test_split(xscale, yscale)

    # fix random seed for reproducibility
    seed = 21
    np.random.seed(seed)

    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=64)

    return(history, X_test, y_test)

def train_w_generator(model, rpath='./', bs=32, epochs=20, scaler=None,
                      n_training=6000,
                      n_validating=500,
                      version=4.0,
                      model_file=None,
                      model_final_file=None,
                      initial_epoch=0,
                      ena_scaling=False):

    if model_file is None:
        model_file = 'denoise_%2.1f' %version

    if model_final_file is None:
        model_final_file = 'denoise_final_%2.1f' %version

    trainGen = data_generator(bs=bs, mode='train', scaler=scaler,
                              rpath=rpath, version=version,
                             ena_scaling=ena_scaling)
    testGen = data_generator(bs=bs, mode='test', scaler=scaler,
                             rpath=rpath, version=version,
                            ena_scaling=ena_scaling)

    # construct the set of callbacks
    bestModel = ModelCheckpoint(model_file,
                                 monitor='val_loss',
                                 mode='min',
                                 verbose=1,
                                 save_best_only=False)

    earlyStop = EarlyStopping(monitor = 'val_loss',
                          min_delta = 0,
                          patience = 5,
                          verbose = 1,
                          restore_best_weights = True)

    # fix random seed for reproducibility
    seed = 21
    np.random.seed(seed)

    history = model.fit(
                        x=trainGen,
                        steps_per_epoch=n_training// bs,
                        validation_data=testGen,
                        validation_steps=n_validating // bs,
                        epochs=epochs,
                        callbacks=[bestModel, earlyStop],
                        initial_epoch=initial_epoch,
                        )

    model.save(model_final_file)

    return(history)

def evaluate(model, X_test, y_test):
    # Final evaluation of the model

    scores = model.evaluate(X_test, y_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1]*100))

def plot_history(history,
                 min_loss=None, max_loss=None,
                 min_metric=None, max_metric=None):

    # Plot history: Loss
    plt.plot(history.history['loss'], label='Training data')
    plt.plot(history.history['val_loss'], label='Validation data')
    plt.title('Activity Loss')
    plt.ylabel('Loss value')
    plt.xlabel('No. epoch')
    plt.legend(loc="upper left")
    plt.ylim(min_loss, max_loss)
    plt.show()

    # Plot history: Accuracy
    plt.plot(history.history['mse'], label='Training mse')
    plt.plot(history.history['val_mse'], label='Validation mse')
    plt.title('Activity MAE')
    plt.ylabel('MAE')
    plt.xlabel('No. epoch')
    plt.legend(loc="upper left")
    plt.ylim(min_metric, max_metric)
    plt.show()
