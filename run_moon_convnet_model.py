################################################
#RUN_MOON_CONVNET_MODEL - for public deployment#
################################################
#This model:
#a) uses a custom loss (separately, i.e. *not* differentiable and guiding backpropagation) to assess how well our algorithm is doing, by connecting the predicted circles to the "ground truth" circles
#b) trained using the original LU78287GT.csv values as the ground truth,
#c) uses the Unet model architechture applied on binary rings.

#This model uses keras version 1.2.2.
############################################

import os
import glob
import numpy as np
import pandas as pd
from PIL import Image

from keras.models import Sequential, Model
from keras.layers.core import Dense, Dropout, Flatten, Reshape
from keras.layers import AveragePooling2D, merge, Input
from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D
from keras.regularizers import l2
from keras.models import load_model

from keras.optimizers import SGD, Adam, RMSprop
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import np_utils
from keras import __version__ as keras_version
from keras import backend as K
K.set_image_dim_ordering('tf')

#custom functions
from utils.rescale_invcolor import rescale_and_invcolor
from utils.template_match_target import template_match_target_to_csv

########################
#custom image generator#
########################################################################
#Following https://github.com/fchollet/keras/issues/2708
def custom_image_generator(data, target, batch_size=32):
    L, W = data[0].shape[0], data[0].shape[1]
    while True:
        for i in range(0, len(data), batch_size):
            d, t = data[i:i+batch_size].copy(), target[i:i+batch_size].copy() #most efficient for memory?
            
            #horizontal/vertical flips
            for j in np.where(np.random.randint(0,2,batch_size)==1)[0]:
                d[j], t[j] = np.fliplr(d[j]), np.fliplr(t[j])               #left/right
            for j in np.where(np.random.randint(0,2,batch_size)==1)[0]:
                d[j], t[j] = np.flipud(d[j]), np.flipud(t[j])               #up/down
            
            #random up/down & left/right pixel shifts, 90 degree rotations
            npix = 15
            h = np.random.randint(-npix,npix+1,batch_size)                  #horizontal shift
            v = np.random.randint(-npix,npix+1,batch_size)                  #vertical shift
            r = np.random.randint(0,4,batch_size)                           #90 degree rotations
            for j in range(batch_size):
                d[j] = np.pad(d[j], ((npix,npix),(npix,npix),(0,0)), mode='constant')[npix+h[j]:L+h[j]+npix,npix+v[j]:W+v[j]+npix,:] #RGB
                t[j] = np.pad(t[j], (npix,), mode='constant')[npix+h[j]:L+h[j]+npix,npix+v[j]:W+v[j]+npix]
                d[j], t[j] = np.rot90(d[j],r[j]), np.rot90(t[j],r[j])
            yield (d, t)

##########################
#unet model (keras 1.2.2)#
########################################################################
#Following https://arxiv.org/pdf/1505.04597.pdf
#and this for merging specifics: https://gist.github.com/Neltherion/f070913fd6284c4a0b60abb86a0cd642
def unet_model(dim,learn_rate,lmbda,FL,init,n_filters):
    print('Making UNET model...')
    img_input = Input(batch_shape=(None, dim, dim, 1))

    a1 = Convolution2D(n_filters, FL, FL, activation='relu', init=init, W_regularizer=l2(lmbda), border_mode='same')(img_input)
    a1 = Convolution2D(n_filters, FL, FL, activation='relu', init=init, W_regularizer=l2(lmbda), border_mode='same')(a1)
    a1P = MaxPooling2D((2, 2), strides=(2, 2))(a1)

    a2 = Convolution2D(n_filters*2, FL, FL, activation='relu', init=init, W_regularizer=l2(lmbda), border_mode='same')(a1P)
    a2 = Convolution2D(n_filters*2, FL, FL, activation='relu', init=init, W_regularizer=l2(lmbda), border_mode='same')(a2)
    a2P = MaxPooling2D((2, 2), strides=(2, 2))(a2)

    a3 = Convolution2D(n_filters*4, FL, FL, activation='relu', init=init, W_regularizer=l2(lmbda), border_mode='same')(a2P)
    a3 = Convolution2D(n_filters*4, FL, FL, activation='relu', init=init, W_regularizer=l2(lmbda), border_mode='same')(a3)
    a3P = MaxPooling2D((2, 2), strides=(2, 2),)(a3)

    u = Convolution2D(n_filters*4, FL, FL, activation='relu', init=init, W_regularizer=l2(lmbda), border_mode='same')(a3P)
    u = Convolution2D(n_filters*4, FL, FL, activation='relu', init=init, W_regularizer=l2(lmbda), border_mode='same')(u)

    u = UpSampling2D((2,2))(u)
    u = merge((a3, u), mode='concat', concat_axis=3)
    u = Convolution2D(n_filters*4, FL, FL, activation='relu', init=init, W_regularizer=l2(lmbda), border_mode='same')(u)
    u = Convolution2D(n_filters*4, FL, FL, activation='relu', init=init, W_regularizer=l2(lmbda), border_mode='same')(u)

    u = UpSampling2D((2,2))(u)
    u = merge((a2, u), mode='concat', concat_axis=3)
    u = Convolution2D(n_filters*2, FL, FL, activation='relu', init=init, W_regularizer=l2(lmbda), border_mode='same')(u)
    u = Convolution2D(n_filters*2, FL, FL, activation='relu', init=init, W_regularizer=l2(lmbda), border_mode='same')(u)

    u = UpSampling2D((2,2))(u)
    u = merge((a1, u), mode='concat', concat_axis=3)
    u = Convolution2D(n_filters, FL, FL, activation='relu', init=init, W_regularizer=l2(lmbda), border_mode='same')(u)
    u = Convolution2D(n_filters, FL, FL, activation='relu', init=init, W_regularizer=l2(lmbda), border_mode='same')(u)

    #final output
    final_activation = 'sigmoid'       #sigmoid, relu
    u = Convolution2D(1, 1, 1, activation=final_activation, init=init, W_regularizer=l2(lmbda), name='output', border_mode='same')(u)
    u = Reshape((dim, dim))(u)
    model = Model(input=img_input, output=u)
    
    #optimizer/compile
    optimizer = Adam(lr=learn_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss='binary_crossentropy', optimizer=optimizer)  #binary cross-entropy severely penalizes opposite predictions.
    print model.summary()

    return model

##################
#Train/Test Model#
########################################################################
#Need to create this function so that memory is released every iteration (when function exits).
#Otherwise the memory used accumulates and eventually the program crashes.
def train_and_test_model(X_train,Y_train,X_valid,Y_valid,X_test,Y_test,loss_data,loss_csvs,dim,learn_rate,nb_epoch,batch_size,save_models,lmbda,FL,init,n_filters):
    model = unet_model(dim,learn_rate,lmbda,FL,init,n_filters)
    
    n_samples = len(X_train)
    for nb in range(nb_epoch):
        model.fit_generator(custom_image_generator(X_train,Y_train,batch_size=batch_size),
                        samples_per_epoch=n_samples,nb_epoch=1,verbose=1,
                        #validation_data=(X_valid, Y_valid), #no generator for validation data
                        validation_data=custom_image_generator(X_valid,Y_valid,batch_size=batch_size),
                        nb_val_samples=n_samples,
                        callbacks=[EarlyStopping(monitor='val_loss', patience=3, verbose=0)])
                        
        # calcualte custom loss
        print ""
        print "custom loss for epoch %d/%d:"%(nb,nb_epoch)
        match_csv_arr, templ_csv_arr, templ_new_arr = [], [], []
        loss_target = model.predict(loss_data.astype('float32'))
        for i in range(len(loss_data)):
            N_match, N_csv, N_templ, csv_duplicate_flag = template_match_target_to_csv(loss_target[i], loss_csvs[i])
            match_csv, templ_csv, templ_new = 0, 0, 0
            if N_csv > 0:
                match_csv = float(N_match)/float(N_csv)             #recall
                templ_csv = float(N_templ)/float(N_csv)             #craters detected/craters in csv
            if N_templ > 0:
                templ_new = float(N_templ - N_match)/float(N_templ) #fraction of craters that are new
            match_csv_arr.append(match_csv); templ_csv_arr.append(templ_csv); templ_new_arr.append(templ_new)
        print "mean and std of N_match/N_csv (recall) = %f, %f"%(np.mean(match_csv_arr), np.std(match_csv_arr))
        print "mean and std of N_template/N_csv = %f, %f"%(np.mean(templ_csv_arr), np.std(templ_csv_arr))
        print "mean and std of (N_template - N_match)/N_template (fraction of craters that are new) = %f, %f"%(np.mean(templ_new_arr), np.std(templ_new_arr))
        print ""
    
    if save_models == 1:
        model.save('models/run_moon_convnet_model_FL%d_%s.h5'%(FL,init))

    return model.evaluate(X_test.astype('float32'), Y_test.astype('float32'))

##############
#Main Routine#
########################################################################
def run_models(dir,learn_rate,batch_size,nb_epoch,n_train_samples,inv_color,rescale,save_models,filter_length,n_filters,lmbda,init):
    #Static arguments
    dim = 256              #image width/height, assuming square images. Shouldn't change
    
    #Load data
    train_data=np.load('%s/Train_rings/train_data.npy'%dir)[:n_train_samples]
    train_target=np.load('%s/Train_rings/train_target.npy'%dir)[:n_train_samples]
    valid_data=np.load('%s/Dev_rings/dev_data.npy'%dir)[:n_train_samples]
    valid_target=np.load('%s/Dev_rings/dev_target.npy'%dir)[:n_train_samples]
    test_data=np.load('%s/Test_rings/test_data.npy'%dir)[:n_train_samples]
    test_target=np.load('%s/Test_rings/test_target.npy'%dir)[:n_train_samples]
    print "Successfully loaded files locally."

    #prepare images for custom loss
    custom_loss_path = '%s/Dev_rings_for_loss'%dir
    loss_data = np.load('%s/custom_loss_images.npy'%custom_loss_path)
    loss_csvs = np.load('%s/custom_loss_csvs.npy'%custom_loss_path)

    #Invert image colors and rescale pixel values to increase contrast
    if inv_color==1 or rescale==1:
        print "inv_color=%d, rescale=%d, processing data"%(inv_color, rescale)
        train_data = rescale_and_invcolor(train_data, inv_color, rescale)
        valid_data = rescale_and_invcolor(valid_data, inv_color, rescale)
        test_data = rescale_and_invcolor(test_data, inv_color, rescale)
        loss_data = rescale_and_invcolor(loss_data, inv_color, rescale)

    #Iterate
    N_runs = np.min((len(filter_length),len(n_filters),len(lmbda),len(init)))
    for i in range(N_runs):
        I = init[i]
        NF = n_filters[i]
        FL = filter_length[i]
        L = lmbda[i]
        score = train_and_test_model(train_data,train_target,valid_data,valid_target,test_data,test_target,loss_data,loss_csvs,dim,learn_rate,nb_epoch,batch_size,save_models,L,FL,I,NF)
        print '###################################'
        print '##########END_OF_RUN_INFO##########'
        print('\nTest Score is %f \n'%score)
        print 'learning_rate=%e, batch_size=%d, filter_length=%e, n_epoch=%d, n_train_samples=%d, img_dimensions=%d, inv_color=%d, rescale=%d, init=%s, n_filters=%d'%(learn_rate,batch_size,FL,nb_epoch,n_train_samples,dim,inv_color,rescale,I,NF)
        print '###################################'
        print '###################################'

################
#Arguments, Run#
########################################################################
if __name__ == '__main__':
    print('Keras version: {}'.format(keras_version))
    
    #args
    dir = 'dataset'         #location of Train_rings/, Dev_rings/, Test_rings/, Dev_rings_for_loss/ folders. Don't include final '/' in path
    lr = 0.0001             #learning rate
    bs = 32                 #batch size: smaller values = less memory but less accurate gradient estimate
    epochs = 6              #number of epochs. 1 epoch = forward/back pass through all train data
    n_train = 6016          #number of training samples, needs to be a multiple of batch size. Big memory hog.
    inv_color = 1           #use inverse color
    rescale = 1             #rescale images to increase contrast (still 0-1 normalized)
    save_models = 1         #save models
    
    ########## Parameters to Iterate Over ##########
    filter_length = [3,3]   #See unet model. Filter length used.
    n_filters = [64,64]     #See unet model. Arranging this so that total number of model parameters <~ 10M, otherwise OOM problems
    lmbda = [0,0]           #See unet model. L2 Weight regularization strength (lambda).
    init = ['he_normal', 'he_uniform']  #See unet model. Initialization of weights.
    ########## Parameters to Iterate Over ##########
    
    #run models
    run_models(dir,lr,bs,epochs,n_train,inv_color,rescale,save_models,filter_length,n_filters,lmbda,init)
