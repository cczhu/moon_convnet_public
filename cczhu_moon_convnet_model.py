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
import time
import pickle

# I/O and math stuff
import itertools
import numpy as np
import pandas as pd

# NN and CV stuff
from sklearn.model_selection import train_test_split #, Kfold
from keras import backend as K
K.set_image_dim_ordering('tf')
import keras
import keras.preprocessing.image as kpimg
from keras.preprocessing.image import Iterator as kerasIterator
from keras.callbacks import EarlyStopping #, ModelCheckpoint

# Silly Keras 2 to Keras 1.2.2 conversion thingy
if K.image_dim_ordering() == "th":
    _img_dim_order = "channels_first"
else:
    _img_dim_order = "channels_last"
    
    
################ DATA READ-IN FUNCTIONS ################


def read_and_normalize_data(Xtr, Ytr, Xdev, Ydev, Xte, Yte,
                            normalize=True, invert=False,
                            rescale=False, verbose=True):
    """Reads and returns input data.
    """
    Xtrain, Ytrain = read_and_norm_sub(Xtr, Ytr, normalize)
    Xdev, Ydev = read_and_norm_sub(Xdev, Ydev, normalize)
    Xtest, Ytest = read_and_norm_sub(Xte, Yte, normalize)
    if verbose:
        print("Loaded data.  N_samples: train = {0}; dev = {1}"
              "test = {2}".format(Xtrain.shape[0], Xdev.shape[0],
                                  Xtest.shape[0]))
        print("Image shapes: X = {0}, Y = {1}".format(Xtrain.shape[1:],
                                                      Ytrain.shape[1:]))
    return Xtrain, Ytrain, Xdev, Ydev, Xtest, Ytest

# read_and_norm_sub globals; change if you feel like you need to customize
# brightness rescaling.
_rescale_low = 0.1
_rescale_high = 1.

def read_and_norm_sub(Xpath, Ypath, normalize, invert, rescale):
    """Sub-function of read_and_normalize_data.  Loads individual data/target
    sets.  Rescaling and inverting intensity based off of rescale_and_invcolor
    from Ari's repo.
    
    Invert and rescale will only work if normalize == True.
    """
    X = np.load(Xpath)
    Y = np.load(Ypath)

    # Rescale X luminosity to between 0 - 1.
    if normalize:
        X /= 255.
        
        if invert:
            X[X > 0.] = 1. - X[X > 0.]

        if rescale:
            for i in range(X.shape[0]):
                # X[i, X[i] > 0] returns view of X!
                CX = X[i, X[i] > 0]
                minv = np.min(CX)
                maxv = np.max(CX)
                CX = _rescale_low + (CX - minv) * \
                    (_rescale_high - _rescale_low) / (maxv - minv)

    # If data doesn't already have a channels axis, add one.
    if len(X.shape) == 3:
        if _img_dim_order == "channels_first":
            X = X.reshape(X.shape[0], 1, *X.shape[1:])
        else:
            X = X.reshape(*X.shape, 1)

    return X, Y


################ MOON DATA ITERATOR ################


class CraterImageGen(object):
    """Heavily modified version of keras.preprocessing.image.ImageDataGenerator
    that performs data augmentation on crater image and ring mask data.

    Parameters
    ----------
    rotation: bool
        If true, randomly rotates by 90, 180 or 270 degrees
    arbitrary_rotation_range: float (0 to 180)
        Range of possible rotations (from -rotation_range to
        +rotation_range).
    offset_range: float (0 to 1)
        +/- fraction of total width and height that can be shifted.
    horizontal_flip: bool
        If True, randomly flip images horizontally.
    vertical_flip: bool
        If True, randomly flip images vertically.
    shear_range: float
        Shear intensity (shear angle in degrees).
    fill_mode: 'constant', 'nearest', 'reflect', or 'wrap'
        Points outside the boundaries are filled according to the
        given mode.  Default is 'constant', which works with default
        cval=0 to keep regions outside border zeroed.
    cval: float
        Value used for points outside the boundaries when fill_mode is
        'constant'. Default is 0.
    contrast_range: float or list
        Amount of random contrast rescaling. if scalar c >= 0, contrast will
        be randomly picked in the range [1-c, 1+c]. A list of [min, max] can be
        passed instead to select this range.  This can be used to
        systematically alter the contrast by passing a list of two identical
        values.  This is more straightforwardly done, however, in
        read_and_normalize_data, where we can rescale all images to have
        a brightness range between user-defined min and max values.
    contrast_keep_mean: bool
        Rescales brightness of image following contrast rescaling so
        that mean brightness remains unchanged from original.
    random_colour_inversion : bool
        Randomly inverts (non-zero) brightnesses.  For inverted images,
        craters are brighter than surroundings.  To systematically do this
        for every image, use the invert option in read_and_normalize_data.
    zoom_range
    data_format: 'channels_first' or 'channels_last'
        In 'channels_first' mode, the channels dimension (the depth)
        is at index 1, in 'channels_last' mode it is at index 3.
        It defaults to the `image_data_format` value found in your
        Keras config file at `~/.keras/keras.json`.
        If you never set it, then it will be 'channels_last'.
    """
    def __init__(self,
                 rotation=True,
                 arbitrary_rotation_range=0.,
                 offset_range=0.,
                 horizontal_flip=True,
                 vertical_flip=True,
                 shear_range=0.,
                 zoom_range=0.,
                 fill_mode='constant',
                 cval=0.,
                 contrast_range=0.,
                 contrast_keep_mean=False,
                 random_colour_inversion=False,
                 data_format=None):

        if data_format is None:
            data_format = _img_dim_order
        self.rotation = rotation
        self.arbitrary_rotation_range = arbitrary_rotation_range
        self.offset_range = offset_range
        self.shear_range = shear_range
        self.fill_mode = fill_mode
        self.cval = cval
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip

        if np.isscalar(zoom_range):
            self.zoom_range = [1 - zoom_range, 1 + zoom_range]
        elif len(zoom_range) == 2:
            self.zoom_range = [zoom_range[0], zoom_range[1]]
        else:
            raise ValueError('`zoom_range` should be a float or '
                             'a tuple or list of two floats. '
                             'Received arg: ', zoom_range)

        if np.isscalar(contrast_range):
            self.contrast_range = [1 - contrast_range, 1 + contrast_range]
        elif len(contrast_range) == 2:
            self.contrast_range = [contrast_range[0], contrast_range[1]]
        else:
            raise ValueError('contrast_range should be a float or '
                             'a tuple or list of two floats. '
                             'Received arg: ', contrast_range)
        self.contrast_keep_mean = contrast_keep_mean
        
        self.random_colour_inversion = random_colour_inversion

        if data_format not in {'channels_last', 'channels_first'}:
            raise ValueError("data_format should be 'channels_last'"
                             "(channel after row and column) or "
                             "'channels_first' (channel before row and "
                             "column).  Received arg: {0}".format(data_format))
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
        """Returns iterator used in training.
        
        Parameters
        ----------
        X : numpy.ndarray
            Source images.  Code expects 4-dimensional tensors with channel
            axis consistent with Keras's image dimension order.
        Y : numpy.ndarray
            Target ring masks.  Code exects 3-dimensional tensors.
        batch_size : int
            Batch size to return each iteration.
        shuffle : bool
            Shuffle data.  Defaults to true.
        seed : None or int
            
        """
        return CraterIterator(
            x, y, self,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
            data_format=self.data_format)

    def standardize(self, x):
        """Apply the standardization configuration to a batch of inputs.
        Currently deals with contrast shift and mean renormalization, and
        intensity inversion.
        
        Parameters
        ----------
        x : numpy.ndarray
            Batch of normalized input to be standardized
            
        Returns
        -------
        numpy.ndarray
            The inputs, standardized.
        """
        if self.contrast_range[0] != 1 and self.contrast_range[1] != 1:

            # If we want to keep the mean brightness same, calculate
            # current mean
            if self.contrast_keep_mean:
                x_mean = np.mean(x)

            # Randomly increase/decrease contrast
            rescale = np.random.uniform(self.contrast_range[0],
                                        self.contrast_range[1])
            x *= rescale
            
            # If we want to keep the mean brightness same, subtract 
            # difference between current and previous mean
            if self.contrast_keep_mean:
                x -= (np.mean(x) - x_mean)

            # Deal with clipping
            x[x > 1.] = 1.
            x[x < 0.] = 0.
            
        if self.random_colour_inversion:
            if np.random.random() < 0.5:
                x[x > 0.] = 1. - x[x > 0.]

        return x

    def random_transform(self, x, y):
        """Randomly augment a single image tensor.  Does NOT copy tensor (this
        is done in CraterIterator.next)

        Parameters
        ----------
        x : numpy.ndarray
            3D tensor, single input image.
        y : numpy.ndarray
            2D tensor, single output target.

        Returns
        -------
            Randomly transformed version of the input and output (same shape).
        """
        # x is a single image, so it doesn't have image number at index 0
        img_row_axis = self.row_axis - 1
        img_col_axis = self.col_axis - 1
        img_channel_axis = self.channel_axis - 1

        # Use composition of homographies to generate final transform
        # that needs to be applied.  From ImageDataGenerator.random_transform
        if self.arbitrary_rotation_range:
            theta = np.pi / 180 * np.random.uniform(
                                        -self.arbitrary_rotation_range,
                                        self.arbitrary_rotation_range)
        else:
            theta = 0

        if self.offset_range:
            tx = np.random.uniform(-self.offset_range,
                                   self.offset_range) * \
                                   x.shape[img_row_axis]
            ty = np.random.uniform(-self.offset_range,
                                   self.offset_range) * \
                                   x.shape[img_col_axis]
        else:
            tx = 0
            ty = 0

        if self.shear_range:
            shear = np.pi / 180. * np.random.uniform(-self.shear_range,
                                                     self.shear_range)
        else:
            shear = 0

        if self.zoom_range[0] == 1 and self.zoom_range[1] == 1:
            zx, zy = 1, 1
        else:
            # Set zx, zy to the same value so aspect ratio is unaltered.
            zx, zy = [np.random.uniform(self.zoom_range[0],
                                       self.zoom_range[1])] * 2

        transform_matrix = None
        if theta != 0:
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                        [np.sin(theta), np.cos(theta), 0],
                                        [0, 0, 1]])
            transform_matrix = rotation_matrix

        if tx != 0 or ty != 0:
            shift_matrix = np.array([[1, 0, tx],
                                     [0, 1, ty],
                                     [0, 0, 1]])
            transform_matrix = shift_matrix if transform_matrix is None \
                else np.dot(transform_matrix, shift_matrix)

        if shear != 0:
            shear_matrix = np.array([[1, -np.sin(shear), 0],
                                    [0, np.cos(shear), 0],
                                    [0, 0, 1]])
            transform_matrix = shear_matrix if transform_matrix is None \
                else np.dot(transform_matrix, shear_matrix)

        if zx != 1 or zy != 1:
            zoom_matrix = np.array([[zx, 0, 0],
                                    [0, zy, 0],
                                    [0, 0, 1]])
            transform_matrix = zoom_matrix if transform_matrix is None \
                else np.dot(transform_matrix, zoom_matrix)

        if transform_matrix is not None:
            xh, xw = x.shape[img_row_axis], x.shape[img_col_axis]
            yh, yw = y.shape[img_row_axis], y.shape[img_col_axis]
            # To use kpimg.apply_transform, y must have a fake channel_axis.
            # We make one here.  img_col_axis > img_channel_axis implies
            # channels first.
            if img_col_axis > img_channel_axis:
                ytemp = y.reshape(1, *y.shape)
            else:
                ytemp = y.reshape(*y.shape, 1)
            x_transform_matrix = kpimg.transform_matrix_offset_center(
                    transform_matrix, xh, xw)
            y_transform_matrix = kpimg.transform_matrix_offset_center(
                    transform_matrix, yh, yw)
            x = kpimg.apply_transform(x, x_transform_matrix, img_channel_axis,
                                      fill_mode=self.fill_mode, cval=self.cval)
            ytemp = kpimg.apply_transform(ytemp, y_transform_matrix,
                                          img_channel_axis,
                                          fill_mode=self.fill_mode,
                                          cval=self.cval)
            # Apply changes back to y.
            if img_col_axis > img_channel_axis:
                y = ytemp[0].copy()
            else:
                y = ytemp[...,0].copy()

        if self.rotation:
            k_r = np.random.randint(0, 3)
            x = np.rot90(x, k=k_r)
            y = np.rot90(y, k=k_r)

        if self.horizontal_flip:
            if np.random.random() < 0.5:
                x = kpimg.flip_axis(x, img_col_axis)
                y = kpimg.flip_axis(y, img_col_axis)

        if self.vertical_flip:
            if np.random.random() < 0.5:
                x = kpimg.flip_axis(x, img_row_axis)
                y = kpimg.flip_axis(y, img_row_axis)

        return x, y


class CraterIterator(kerasIterator):
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
            raise ValueError("X (images) and Y (targets) "
                             "should have the same length. "
                             "Found: X.shape = {0}, Y.shape = {1}".format(
                                     x.shape[0], y.shape[0]))

        if data_format is None:
            data_format = _img_dim_order
        self.x = np.asarray(x, dtype=K.floatx())

        if self.x.ndim != 4:
            raise ValueError("Input data in `NumpyArrayIterator` "
                             "should have rank 4. You passed an array "
                             "with shape {0}".format(self.x.shape))
        channels_axis = 3 if data_format == 'channels_last' else 1
        if self.x.shape[channels_axis] not in {1, 3, 4}:
            raise ValueError("NumpyArrayIterator is set to use the data "
                             "format convention {0} (channels on axis {1}),"
                             "i.e. expected either 1, 3 or 4 channels on axis"
                             "{1}. However, it was passed an array with shape "
                             "{2} ({3} channels).".format(
                                     data_format, str(channels_axis),
                                     str(self.x.shape),
                                     str(self.x.shape[channels_axis])))

        if y is not None:
            self.y = np.asarray(y, dtype=K.floatx())
        else:
            self.y = None

        self.image_data_generator = image_data_generator
        self.data_format = data_format
        super(CraterIterator, self).__init__(x.shape[0], batch_size,
                                             shuffle, seed)

    def next(self):
        """For Python 2.x. (Python 3.x uses Iterator.__next__, which 
        maps back to this function.)  Returns the next batch.
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

            # "copy = True" is the default behaviour of astype, but
            # I made this explicit to make this readable
            x, y = self.image_data_generator.random_transform(
                                x.astype(K.floatx(), copy=True),
                                y.astype(K.floatx(), copy=True))
            x = self.image_data_generator.standardize(x)

            batch_x[i] = x
            batch_y[i] = y

        return batch_x, batch_y


################ TRAINING ROUTINE ################


def train_test_model(Xtrain, Ytrain, Xtest, Ytest, lambd, args):

    Xtr, Xval, Ytr, Yval = train_test_split(Xtrain, Ytrain, test_size=args['test_size'], 
                                                    random_state=args['random_state'])
    gen = MoonImageGen(width_shift_range=1./args['imgshp'][1],
                         height_shift_range=1./args['imgshp'][0],
                         fill_mode='constant',
                         horizontal_flip=True, vertical_flip=True)

    model = cczhu_cnn(args)
    model.fit_generator( gen.flow(Xtr, Ytr, batch_size=args['batchsize'], shuffle=True),
                        samples_per_epoch=len(Xtr), nb_epoch=args['N_epochs'], 
                        validation_data=gen.flow(Xval, Yval, batch_size=args['batchsize']),
                        nb_val_samples=len(Xval), verbose=1,
                        callbacks=[EarlyStopping(monitor='val_loss', patience=3, verbose=0)])

    #model_name = ''
    #model.save_weights(model_name)     #save weights of the model
     
    test_predictions = model.predict(test_data.astype('float32'), batch_size=batch_size, verbose=2)
    return mean_absolute_error(Ytest, Ypred)  #calculate test score


################ MAIN ACCESS ROUTINE ################

def run_model(in_args, verbose=True, out_csv=False):
    
    # Read in data, normalizing input images
    Xtrain, Ytrain, Xdev, Ydev, Xtest, Ytest = read_and_normalize_data(
                                        in_args.Xtrainpath, in_args.Ytrainapth,
                                        in_args.Xdevpath, in_args.Ydevpath,
                                        in_args.Xtestpath, in_args.Ytestpath,
                                        normalize=True)

    columnvals = ['Learn Rate', 'Batch', 'Epochs', 'N_train', 'Img Dim'] + \
        list(in_args.hyper_table.columns)
    
    if out_csv:
        outcsv = open(in_args + ".csv", 'rw')
        outcsv.write(", ".join(columnvals))

    for i in range(len(in_args.hyper_table)):
        hargs, test_score = train_test_model(Xtrain, Ytrain, Xtest, 
                                             Ytest, in_args, i)
        csvstr = in_args.get_csv_str(**hargs)
        if out_csv:
            outcsv.write(", ".join(csvstr))
        if verbose:
            print("#####################################")
            print("########## END OF RUN INFO ##########")
            print("\nTest Score: {0:e}\n".format(test_score))
            print("\tFor")
            print("\t".join(columnvals))
            print("\t".join(csvstr))
            print("#####################################")
            print("#####################################")


################ MAIN ACCESS ROUTINE INPUT CLASS ################
#https://stackoverflow.com/questions/21527610/pythonic-way-to-pass-around-many-arguments


class ConvnetInputs(object):
    """
    Stores input parameters (including private ones that are set by the CNN
    architecture), and generates derived parameters to use in run functions.
    Also generates table of

    Parameters
    ----------
    filedir : str
        Input npy file directory (trailing slash unnecessary)
    lr : float
        Learning rate
    bs : int
        Batch size, must be larger than 1
    epochs : int
        Number of training epochs
    n_train : int
        Number of training samples, must be multiple of bs
    gen_args : dict
        Args to image generator.  See MoonIterator docstring for details.
    autotable : bool
        Automatically generate hyperparameter table for looping.  Default
        is True.
    table_args : dict
        If ``autotable == True``, dict passed to automatic hyperparameter
        table generator, which generates
    save_models : bool
        If true, saves models to save_prefix directory
    save_prefix : str
        If ``save_models == True``, path and file prefix used for saves.
    """

    def __init__(self, filedir, lr, bs, epochs, n_train,
                 gen_args, autotable=True, table_args={},
                 save_models=True, save_prefix="./models/run"):

        self.filedir = filedir
        self._train_data = "/Train_rings/train_data.npy"

        self.lr = lr
        self.bs = bs
        self.epochs = epochs
        self.n_train = n_train

        self.gen_args = gen_args

        if autotable:
            self.table_generator(table_args)
        else:
            self.hyper_table = None
            self._table_raw = None

        self.save_models = save_models
        self.save_prefix = save_prefix

        # Static arguments (dependent on CNN structure)
        self._img_dim = 256

    @property
    def epochs(self):
        return self._epochs

    @epochs.setter
    def epochs(self, value):
        assert value >= 1, "Must train for more than 1 epoch!"
        self._epochs = value

    @property
    def bs(self):
        return self._bs

    @bs.setter
    def bs(self, value):
        assert value >= 1, "Batch size must be greater than 1!"
        self._bs = value

    @property
    def n_train(self):
        return self._n_train

    @n_train.setter
    def n_train(self, value):
        assert value % self.bs == 0, "n_train must be divisible by bs!"
        self._n_train = value

    @property
    def img_dim(self):
        return self._img_dim

    @property
    def Xtrainpath(self):
        return self.filedir + "/Train_rings/train_data.npy"

    @property
    def Ytrainpath(self):
        return self.filedir + "/Train_rings/train_target.npy"

    @property
    def Xdevpath(self):
        return self.filedir + "/Dev_rings/dev_data.npy"

    @property
    def Ydevpath(self):
        return self.filedir + "/Dev_rings/dev_target.npy"

    @property
    def Xtestpath(self):
        return self.filedir + "/Test_rings/test_data.npy"

    @property
    def Ytestpath(self):
        return self.filedir + "/Test_rings/test_target.npy"

    # Specifically not using pandas to reduce number of dependencies
    def table_generator(self, targs, randomise=False):
        """
        Auto hyperparameter table generator.  Currently able to modify:

            filter_length
            n_filters
            lmbda
            weight_init

        To specify a hyperparameter range, include in targs an entry with
        the corresponding hyperparameter name and a four-tuple

            targs[name] = (min, max, num, lin/log/log10)

        where min and max are the minimum and maximum possible values, num
        is the number of values, and lin/log should either be "lin", "log" or
        "log10" for linear or logarithmic (e or 10) scaling.  The function will
        then pass the values to

            np.linspace(min, max, num)
                or
            10**np.linspace(np.log10(min), np.log10(max), num)

        Alternatively, a list of possible values a hyperparameter can take
        can be passed, e.g.

            targs['filter_lengths'] = [3, 5, 7]

        This will be directly used without modification.

        If necessary parameters are not included, function falls back on
        defaults.  Function ignores any unrecognized parameters passed.

        Parameters
        ----------
        targs : dict
            Dict of table arguments.
        randomise : bool
            If True, randomise rows.
        """
        hyper_table_columns = ['filter_length', 'n_filters',
                               'lmbda', 'weight_init']
        table_defaults = dict(zip(hyper_table_columns,
                                  [3, 64, 0., 'he_normal']))

        # Copy for internal use, then fill in missing hyperparams
        args = targs.copy()
        for key in hyper_table_columns:
            if key not in args.keys():
                args[key] = [table_defaults[key]]

        # Tuple of range values for each hyperparameter
        self._table_raw = tuple(args[item] if isinstance(args[item], list) else
                                self.table_subloop(args[item]) for
                                item in hyper_table_columns)

        # Generate table
        self.hyper_table = pd.DataFrame(
                                list(itertools.product(*self._table_raw)),
                                columns=hyper_table_columns)

        # https://stackoverflow.com/questions/29576430/shuffle-dataframe-rows
        if randomise:
            self.hyper_table = self.hyper_table.sample(frac=1)

    @staticmethod
    def table_subloop(param):
        """Subloop for pre-processing hyperparameters before tabulation.
        """
        if not isinstance(param, tuple):
            return list([param])
        assert len(param) == 4, "Must have four elements in tuple!"

        if param[3] == 'log':
            return np.exp(np.linspace(np.log(param[0]),
                                      np.log(param[1]),
                                      param[2]))
        if param[3] == 'log10':
            return 10**np.linspace(np.log10(param[0]),
                                   np.log10(param[1]),
                                   param[2])
        return np.linspace(param[0], param[1], param[2])

    def get_csv_str(self, **kwargs):
        """Returns a csv row for CNN run parameters.
        """
        printstr = ', '.join([
                        "{0:.4e}".format(self.lr),
                        "{0:d}".format(self.bs),
                        "{0:d}".format(self.epochs),
                        "{0:d}".format(self.n_train),
                        "{0:d}".format(self.img_dim)])
        if len(kwargs):
            printstr += ", "
            for key in kwargs.keys():
                printstr += "{0}, ".format(kwargs[key])
            printstr = printstr[:-1]  # Remove last comma

        return printstr

    def print_everything(self):
        """Outputs all relevant parameters input parameters.
        """
        vals = self.get_csv_str().split(',')
        name = "< " + self.__class__.__name__ + " "
        printstr = name + ("\n" + len(name) * " ").join(
                ["Learning rate = {0}".format(vals[0]),
                 "Batch size = {0}".format(vals[1]),
                 "Epochs = {0}".format(vals[2]),
                 "N_train = {0}".format(vals[3]),
                 "Img dim = {0}".format(vals[4])])

        if len(self.gen_args):
            printstr += "\n" + len(name) * " " + "Generator Args:"
            for key in self.gen_args.keys():
                printstr += "\n" + (len(name) + 4) * " " + \
                            "{0:s} = {1:s}".format(key,
                                                     str(self.gen_args[key]))

        if isinstance(self.hyper_table, pd.DataFrame):
            printstr += "\n" + len(name) * " " + "Table Raw Args: "
            for i, key in enumerate(self.hyper_table.columns):
                printstr += "\n" + (len(name) + 4) * " " + \
                            "{0}: {1}".format(
                                key, list(self.hyper_table[key].unique()))
        printstr += " >"

        return printstr

    def __repr__(self):
        return self.print_everything()
