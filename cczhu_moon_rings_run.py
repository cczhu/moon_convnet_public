import sys
from keras import __version__ as keras_version
from keras import backend as K
K.set_image_dim_ordering('tf')

sys.path.append('/home/m/mhvk/czhu/moon_convnet_public/')
import run_moon_convnet_model as model
 
#args
dir = './dataset'         #location of Train_rings/, Dev_rings/, Test_rings/, Dev_rings_for_loss/ folders. Don't include final '/' in path
lr = 0.0001             #learning rate
bs = 32                 #batch size: smaller values = less memory but less accurate gradient estimate
epochs = 6              #number of epochs. 1 epoch = forward/back pass through all train data
n_train = 6016          #number of training samples, needs to be a multiple of batch size. Big memory hog.
inv_color = 1           #use inverse color
rescale = 1             #rescale images to increase contrast (still 0-1 normalized)
save_models = 1         #save models

########## Parameters to Iterate Over ##########
filter_length = [3,3,5]   #See unet model. Filter length used.
n_filters = [64,64,64]     #See unet model. Arranging this so that total number of model parameters <~ 10M, otherwise OOM problems
lmbda = [0,0,0]           #See unet model. L2 Weight regularization strength (lambda).
init = ['he_normal', 'he_uniform', 'he_normal']  #See unet model. Initialization of weights.
########## Parameters to Iterate Over ##########

#run models
model.run_models(dir,lr,bs,epochs,n_train,inv_color,rescale,save_models,filter_length,n_filters,lmbda,init)
