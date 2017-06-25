import glob
import numpy as np

from keras.models import load_model
from keras import backend as K
from utils.helper_functions import rescale_and_invcolor

##############
#Main Routine#
########################################################################
def predict_targets(dir,inv_color,rescale,n_pred_samples,offset,models):
    #static arguments
    dim = 256               #image dimensions, assuming square images. Should not change
    
    #load data
    if n_pred_samples < 50:
        try:
            test_data = np.load('%s/Test_rings/test_data_50im.npy'%dir)[:n_pred_samples]
            test_target = np.load('%s/Test_rings/test_target_50im.npy'%dir)[:n_pred_samples]
        except:
            print "Couldn't find 50 image subset numpy arrays. Loading full data. Saving subset of 50 images for future use."
            test_data = np.load('%s/Test_rings/test_data.npy'%dir)[:n_pred_samples]
            test_target = np.load('%s/Test_rings/test_target.npy'%dir)[:n_pred_samples]
            np.save('%s/Test_rings/test_data_50im.npy'%dir,test_data[0:50])
            np.save('%s/Test_rings/test_target_50im.npy'%dir,test_target[0:50])
    else:
        test_data = np.load('%s/Test_rings/test_data.npy'%dir)[:n_pred_samples]
        test_target = np.load('%s/Test_rings/test_target.npy'%dir)[:n_pred_samples]
    test_data = rescale_and_invcolor(test_data, inv_color, rescale)

    print "begin generating predictions"
    for m in models:
        model = load_model('models/%s'%m)
        target_pred = model.predict(test_data[offset:(n_pred_samples+offset)].astype('float32'))
        
        #dimensions go data, ground_truth targets, predicted targets
        result = np.concatenate((test_data[offset:(n_pred_samples+offset)],
                                 test_target[offset:(n_pred_samples+offset)].reshape(n_pred_samples,dim,dim,1),
                                 target_pred.reshape(n_pred_samples,dim,dim,1)),axis=3)
                                 
        name = m.split('.h5')[0]
        np.save('models/%s_pred.npy'%name,result)
        print "successfully generated predictions at models/%s_pred.npy for model %s"%(name,m)

################
#Arguments, Run#
########################################################################
if __name__ == '__main__':
    #arguments
    dir = 'dataset'         #location of where test data is. Likely doesn't need to change
    inv_color = 1           #use inverse color (*must be same setting as what was used for the model(s)*)
    rescale = 1             #rescale images to increase contrast (*must be same setting as what was used for the model(s)*)
    n_pred_samples = 20     #number of test images to predict on
    offset = 0              #index offset to start predictions at in test array
    models = ['models/unet_s256_rings_FL3_he_normal_customloss.h5']
    
    predict_targets(dir,inv_color,rescale,n_pred_samples,offset,models)
    

