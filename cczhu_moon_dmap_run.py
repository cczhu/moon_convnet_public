"""
Charles's attempt at creating an I/O pipeline and convnet for crater counting.
"""

################ IMPORTS ################


# Past-proofing
from __future__ import absolute_import, division, print_function

# System modules
import os
import sys
import glob
import cv2
import datetime
import pickle

# I/O and math stuff
import pandas as pd
import numpy as np

sys.path.append("/home/m/mhvk/czhu/moon_craters")
import make_density_map as densmap

# NN and CV stuff
from sklearn.model_selection import train_test_split #, Kfold
from keras import backend as K
K.set_image_dim_ordering('tf')
import keras
import keras.preprocessing.image as kpimg
from keras.preprocessing.image import Iterator
from keras.callbacks import EarlyStopping #, ModelCheckpoint

# Silly Keras 2 to Keras 1.2.2 conversion thingy
def get_image_data_format():
    img_dim_order = K.image_dim_ordering()
    if img_dim_order == "th":
        return "channels_first"
    return "channels_last"

################ DATA READ-IN FUNCTIONS ################


def read_and_normalize_data(path, Xtr, Ytr, Xte, Yte, normalize=True):
    """Reads and returns input data.
    """
    Xtrain = np.load(path + Xtr)
    Ytrain = np.load(path + Ytr)
    Xtest = np.load(path + Xte)
    Ytest = np.load(path + Yte)
    if normalize:
        Xtrain /= 255.
        Xtest /= 255.
    print("Loaded data.  N_samples: train = "
          "{0:d}; test = {1:d}".format(Xtrain.shape[0], Xtest.shape[0]))
    print("Image shapes: X =", Xtrain.shape[1:], \
                                "Y = {1:d}", Ytrain.shape[1:])
    return Xtrain, Ytrain, Xtest, Ytest



################ MOON DATA ITERATOR ################


class MoonImageGen(object)
    """Heavily modified version of keras.preprocessing.image.ImageDataGenerator.
    that creates density maps or masks, and performs random transformation
    on both source image and result consistently by treating the map/mask
    as another colour channel.

    Parameters
    ----------
    rotation_range: float (0 to 180)
        Range of possible rotations (from -rotation_range to 
        +rotation_range).
    width_shift_range: float (0 to 1)
        +/- fraction of total width that can be shifted.
    height_shift_range: float (0 to 1)
        +/- fraction of total height that can be shifted.
    shear_range: float
        Shear intensity (shear angle in radians).
    fill_mode: points outside the boundaries are filled according to the
        given mode ('constant', 'nearest', 'reflect' or 'wrap'). Default
        is 'constant', which works with default cval=0 to keep regions
        outside border zeroed.
    cval: value used for points outside the boundaries when fill_mode is
        'constant'. Default is 0.
    horizontal_flip: whether to randomly flip images horizontally.
    vertical_flip: whether to randomly flip images vertically.
    contrast_range: amount of random contrast rescaling. if scalar c >= 0, contrast will be 
            randomly picked in the range [1-c, 1+c]. A sequence of [min, max] can be 
            passed instead to select this range.
    contrast_keep_mean: rescales brightness of image following contrast rescaling so
            that mean brightness remains unchanged from original.
    data_format: 'channels_first' or 'channels_last'. In 'channels_first' mode, the channels dimension
        (the depth) is at index 1, in 'channels_last' mode it is at index 3.
        It defaults to the `image_data_format` value found in your
        Keras config file at `~/.keras/keras.json`.
        If you never set it, then it will be "channels_last".
    """
    def __init__(self,
                 rotation_range=0.,
                 #width_shift_range=0.,
                 #height_shift_range=0.,
                 shear_range=0.,
                 fill_mode='constant',
                 cval=0.,
                 horizontal_flip=False,
                 vertical_flip=False,
                 contrast_range=0.,
                 contrast_keep_mean=False,
                 data_format=None):

        if data_format is None:
            data_format = get_image_data_format()
        self.rotation_range = rotation_range
        #self.width_shift_range = width_shift_range
        #self.height_shift_range = height_shift_range
        self.shear_range = shear_range
        self.channel_shift_range = channel_shift_range
        self.fill_mode = fill_mode
        self.cval = cval
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip

        if np.isscalar(contrast_range):
            self.contrast_range = [1 - contrast_range, 1 + contrast_range]
        elif len(contrast_range) == 2:
            self.contrast_range = [contrast_range[0], contrast_range[1]]
        else:
            raise ValueError('contrast_range should be a float or '
                             'a tuple or list of two floats. '
                             'Received arg: ', contrast_range)

        self.contrast_keep_mean = contrast_keep_mean

        if data_format not in {'channels_last', 'channels_first'}:
            raise ValueError('data_format should be "channels_last" (channel after row and '
                             'column) or "channels_first" (channel before row and column). '
                             'Received arg: ', data_format)
        self.data_format = data_format
        if data_format == 'channels_first':
            self.channel_axis = 1
            self.row_axis = 2
            self.col_axis = 3
        if data_format == 'channels_last':
            self.channel_axis = 3
            self.row_axis = 1
            self.col_axis = 2


    def flow(self, x, y, batch_size=32, shuffle=True, seed=None):
        return MoonIterator(
            x, y, self,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
            data_format=self.data_format)


    def standardize(self, x):
        """Apply the standardization configuration to a batch of inputs.
        Currently only deals with contrast shift and mean renormalization.
        # Arguments
            x: batch of normalized input to be standardized
        # Returns
            The inputs, standardized.
        """
        if self.contrast_range[0] != 1 and self.contrast_range[1] != 1:

            # If we want to keep the mean brightness same, calculate
            # current mean
            if self.contrast_keep_mean:
                x_mean = np.mean(x)

            # Randomly increase/decrease contrast
            rescale = np.random.uniform(self.zoom_range[0], self.zoom_range[1])
            x *= rescale
            
            # If we want to keep the mean brightness same, subtract 
            # difference between current and previous mean
            if self.contrast_keep_mean:
                x -= (np.mean(x) - x_mean)

            # Deal with clipping
            x[x > 1.] = 1.
            x[x < 0.] = 0.

        return x


    def random_transform(self, x, y):
        """Randomly augment a single image tensor.
        # Arguments
            x: 2D tensor, single input image.
            y: 2D tensor, single output target.
        # Returns
            A randomly transformed version of the input and output (same shape).
        """
        # x is a single image, so it doesn't have image number at index 0
        img_row_axis = self.row_axis - 1
        img_col_axis = self.col_axis - 1
        img_channel_axis = self.channel_axis - 1

        # use composition of homographies
        # to generate final transform that needs to be applied
        if self.rotation_range:
            theta = np.pi / 180 * np.random.uniform(-self.rotation_range, self.rotation_range)
        else:
            theta = 0

        #if self.height_shift_range:
        #    tx = np.random.uniform(-self.height_shift_range, self.height_shift_range) * x.shape[img_row_axis]
        #else:
        #    tx = 0

        #if self.width_shift_range:
        #    ty = np.random.uniform(-self.width_shift_range, self.width_shift_range) * x.shape[img_col_axis]
        #else:
        #    ty = 0

        if self.shear_range:
            shear = np.random.uniform(-self.shear_range, self.shear_range)
        else:
            shear = 0

        #if self.zoom_range[0] == 1 and self.zoom_range[1] == 1:
        #    zx, zy = 1, 1
        #else:
        #    zx, zy = np.random.uniform(self.zoom_range[0], self.zoom_range[1], 2)

        transform_matrix = None
        if theta != 0:
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                        [np.sin(theta), np.cos(theta), 0],
                                        [0, 0, 1]])
            transform_matrix = rotation_matrix

        #if tx != 0 or ty != 0:
        #    shift_matrix = np.array([[1, 0, tx],
        #                             [0, 1, ty],
        #                             [0, 0, 1]])
        #    transform_matrix = shift_matrix if transform_matrix is None else np.dot(transform_matrix, shift_matrix)

        if shear != 0:
            shear_matrix = np.array([[1, -np.sin(shear), 0],
                                    [0, np.cos(shear), 0],
                                    [0, 0, 1]])
            transform_matrix = shear_matrix if transform_matrix is None else np.dot(transform_matrix, shear_matrix)

        #if zx != 1 or zy != 1:
        #    zoom_matrix = np.array([[zx, 0, 0],
        #                            [0, zy, 0],
        #                            [0, 0, 1]])
        #    transform_matrix = zoom_matrix if transform_matrix is None else np.dot(transform_matrix, zoom_matrix)

        if transform_matrix is not None:
            xh, xw = x.shape[img_row_axis], x.shape[img_col_axis]
            yh, yw = y.shape[img_row_axis], y.shape[img_col_axis]
            x_transform_matrix = transform_matrix_offset_center(transform_matrix, xh, xw)
            y_transform_matrix = transform_matrix_offset_center(transform_matrix, yh, yw)
            x = apply_transform(x, x_transform_matrix, img_channel_axis,
                                fill_mode=self.fill_mode, cval=self.cval)
            y = apply_transform(y, y_transform_matrix, img_channel_axis,
                                fill_mode=self.fill_mode, cval=self.cval)

        if self.horizontal_flip:
            if np.random.random() < 0.5:
                x = flip_axis(x, img_col_axis)
                y = flip_axis(y, img_col_axis)

        if self.vertical_flip:
            if np.random.random() < 0.5:
                x = flip_axis(x, img_row_axis)
                y = flip_axis(y, img_row_axis)

        return x, y


class MoonIterator(Iterator):
    """Iterator yielding input crater images and output targets.

    Parameters
    ----------
    x: numpy.array
        3D numpy array of input data.
    y: numpy.array
        3D numpy array of output targets.
    moon_image_gen: MoonImageGen instance
        Instance to use for random transformations 
        and standardization.
    batch_size: int
        Size of a batch.
    shuffle: bool
        Toggle whether to shuffle the data between epochs.
    seed: int
        Random seed for data shuffling.
    data_format: str
        One of `channels_first`, `channels_last`.  If None,
        obtains value from Keras backend
    """

    def __init__(self, x, y, image_data_generator,
                 batch_size=32, shuffle=False, seed=None,
                 data_format=None):

        if y is not None and len(x) != len(y):
            raise ValueError('X (images) and Y (targets) '
                             'should have the same length. '
                             'Found: X.shape = %s, Y.shape = %s' %
                             (x.shape[0], y.shape[0]))

        if data_format is None:
            data_format = get_image_data_format()
        self.x = np.asarray(x, dtype=K.floatx())

        if self.x.ndim != 4:
            raise ValueError('Input data in `NumpyArrayIterator` '
                             'should have rank 4. You passed an array '
                             'with shape', self.x.shape)
        channels_axis = 3 if data_format == 'channels_last' else 1
        if self.x.shape[channels_axis] not in {1, 3, 4}:
            raise ValueError('NumpyArrayIterator is set to use the '
                             'data format convention "' + data_format + '" '
                             '(channels on axis ' + str(channels_axis) + '), i.e. expected '
                             'either 1, 3 or 4 channels on axis ' + str(channels_axis) + '. '
                             'However, it was passed an array with shape ' + str(self.x.shape) +
                             ' (' + str(self.x.shape[channels_axis]) + ' channels).')

        if y is not None:
            self.y = np.asarray(y, dtype=K.floatx())
        else:
            self.y = None

        self.image_data_generator = image_data_generator
        self.data_format = data_format
        super(MoonIterator, self).__init__(x.shape[0], batch_size, shuffle, seed)


    def next(self):
        """For python 2.x. (python 3.x uses Iterator.__next__, which 
        maps back to this function.
        # Returns
            The next batch.
        """
        # Keeps under lock only the mechanism which advances
        # the indexing of each batch.
        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)
        # The transformation of images is not under thread lock
        # so it can be done in parallel
        batch_x = np.zeros(tuple([current_batch_size] + list(self.x.shape)[1:]), dtype=K.floatx())
        batch_y = np.zeros(tuple([current_batch_size] + list(self.y.shape)[1:]), dtype=K.floatx())
        for i, j in enumerate(index_array):
            x = self.x[j]
            y = self.y[j]

            x, y = self.image_data_generator.random_transform(x.astype(K.floatx()),
                                                              y.astype(K.floatx()))
            x = self.image_data_generator.standardize(x)

            batch_x[i] = x
            batch_y[i] = y

        return batch_x, batch_y


################ TRAINING ROUTINE ################


def train_test_model(Xtrain, Ytrain, Xtest, Ytest, lambd, args):

    Xtr, Xval, Ytr, Yval = train_test_split(Xtrain, Ytrain, test_size=args["test_size"], 
                                                    random_state=args["random_state"])
    gen = MoonDataGenerator(width_shift_range=1./args["imgshp"][1],
                         height_shift_range=1./args["imgshp"][0],
                         fill_mode='constant',
                         horizontal_flip=True, vertical_flip=True)

    model = cczhu_cnn(args)
    model.fit_generator( gen.flow(Xtr, Ytr, batch_size=args["batchsize"], shuffle=True),
                        samples_per_epoch=len(Xtr), nb_epoch=args["N_epochs"], 
                        validation_data=gen.flow(Xval, Yval, batch_size=args["batchsize"]),
                        nb_val_samples=len(Xval), verbose=1,
                        callbacks=[EarlyStopping(monitor='val_loss', patience=3, verbose=0)])

    #model_name = ''
    #model.save_weights(model_name)     #save weights of the model
     
    test_predictions = model.predict(test_data.astype('float32'), batch_size=batch_size, verbose=2)
    return mean_absolute_error(Ytest, Ypred)  #calculate test score


def print_details(args, lambd):
    print("Learning rate = {0:e}; Batch size = {1:d}, lambda = {2:d}".format( \
            args["learn_rate"], args["batchsize"], lambd))
    print("Number of epochs = {0:d}, N_training_samples = {1:d}".format( \
            args["N_epochs"], args["N_sub_train"]))


def run_cross_validation(Xtrain, Ytrain, Xtest, Ytest, args):

    # Get randomized lambda values for L2 regularization.  Always have 10^0 as baseline.
    lambd_space = np.logspace(args["CV_lambd_range"][0], 
                                    args["CV_lambd_range"][1], 10*args["CV_lambd_N"])
    lambd_arr = np.r_[np.random.choice(lambd_space, args["CV_lambd_N"]),
                    np.array([0]))

    for i, lambd in enumerate(lambd_arr):
        test_score = train_test_model(Xtrain, Ytrain, Xtest, Ytest, lambd, args)
        print('#####################################')
        print('########## END OF RUN INFO ##########')
        print("\nTest Score: {0:e}\n".format(test_score))
        print_details(args, lambd)
        print('#####################################')
        print('#####################################')


################ MAIN ################


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(description='Keras-based CNN for mapping crater images to density maps.')
    parser.add_argument("path", 
    parser.add_argument("Xtrain", type=str, help="Training input npy file")
    parser.add_argument("Xtest", type=str, help="Testing input npy file")
    parser.add_argument("Ytrain", type=str, help="Training target npy file")
    parser.add_argument("Ytest", type=str, help="Testing target npy file")
    parser.add_argument('--learn_rate', type=float, required=False,
                        help='Learning rate', default=0.0001)
#    parser.add_argument('--crater_cutoff', type=int, required=False,
#                        help='Crater pixel diameter cutoff', default=3)
    parser.add_argument('--batchsize', type=int, required=False,
                        help='Crater pixel diameter cutoff', default=32)
    parser.add_argument('--lambd', type=float, required=False,
                        help='L2 regularization coefficient', default=0.)
    parser.add_argument('--epochs', type=int, required=False,
                        help='Number of training epochs', default=30)
    parser.add_argument('--f_samp', type=int, required=False,
                        help='Random fraction of samples to use', default=1.)
    parser.add_argument('--dumpweights', type=str, required=False,
                        help='Filename to dump NN weights to file')
    parser.add_argument('--dumpargs', type=str, required=False,
                        help='Filename to dump arguments into pickle')


#    parser.add_argument('--lu_csv_path', metavar='lupath', type=str, required=False,
#                        help='Path to LU78287 crater csv.', default="./LU78287GT.csv")
#    parser.add_argument('--alan_csv_path', metavar='lupath', type=str, required=False,
#                        help='Path to LROC crater csv.', default="./alanalldata.csv")
#    parser.add_argument('--outhead', metavar='outhead', type=str, required=False,
#                        help='Filepath and filename prefix of outputs.', default="out/lola")
#    parser.add_argument('--amt', type=int, default=7500, required=False,
#                        help='Number of images each thread will make (multiply by number of \
#                        threads for total number of images produced).')

    in_args = parser.parse_args()

    # Print Keras version, just in case
    print('Keras version: {0}'.format(keras.__version__))

    # Read in data, normalizing input images
    Xtrain, Ytrain, Xtest, Ytest = read_and_normalize_data(in_args.path, in_args.Xtrain, 
                                        in_args.Ytrain, in_args.Xtest, in_args.Ytest, 
                                        normalize=True)

    # Declare master dictionary of input variables
    args = {}

    # Get image and target sizes
    args["imgshp"] = tuple(Xtrain.shape[1:])
    args["tgtshp"] = tuple(Ytrain.shape[1:])

    # Load constants from user
    args["path"] = args.path
    args["learn_rate"] = in_args.learning_rate
    #args["c_pix_cut"] = in_args.crater_cutoff

    args["batchsize"] = in_args.batchsize
    args["lambda"] = in_args.lambd
    args["N_epochs"] = in_args.epochs
    args["f_samp"] = in_args.f_samp

    args["dumpweights"] = in_args.dumpweights

    # Calculate next largest multiple of batchsize to N_train*f_samp
    # Then use to obtain subset
    args["N_sub_train"] = int(args["batchsize"] * np.ceil( Xtrain.shape[0] * \
                                        args["f_samp"] / args["batchsize"] ))
    args["sub_train"] = np.random.choice(X.shape[0], size=args["N_sub_train"])
    Xtrain = Xtrain[args["sub_train"]]
    Ytrain = Ytrain[args["sub_train"]]


    # Cross-validation parameters that could be user-selectable at a later date.
    args["CV_lambd_N"] = 10             # Number of L2 regularization coefficients to try
    args["CV_lambd_range"] = (-3, 0)    # Log10 range of L2 regularization coefficients
    args["random_state"] = None         # Random initializer for train_test_split
    args["test_size"] = 0.2             # train_test_split fraction of examples for validation


    # CALL TRAIN TEST

    if in_args.dumpargs:
        pickle.dump( args, open(in_args.dumpargs, 'wb') )
