"""
Convolutional neural network (CNN) solution to crater counting on rocky bodies.
"""

import sys
import numpy as np

sys.path.append('/home/m/mhvk/czhu/moon_convnet_public/')
import cczhu_moon_convnet_model as model

########## Input parameters ##########

filedir = './dataset'   #location of data.  Don't include final '/' in path
lr = 0.0001             #learning rate
bs = 32                 #batch size: smaller values = less memory but less accurate gradient estimate
epochs = 6              #number of epochs. 1 epoch = forward/back pass through all train data
n_train = 6016          #number of training samples, needs to be a multiple of batch size. Big memory hog.
inv_color = 1           #use inverse color
rescale = 1             #rescale images to increase contrast (still 0-1 normalized)
save_models = 1         #save models

########## Hyperparameter range ##########

gen_args = {}

gen_args['horizontal_flip'] = True
gen_args['vertical_flip'] = True
gen_args['contrast_range'] = [1., 1.4]
gen_args['contrast_keep_mean'] = True

table_args = {}
# Filter size.
table_args['filter_length'] = [3, 5]
# Number of filters.  Ensure total number of model  parameters <~ 10M, otherwise OOM problems
table_args['n_filters'] = [64]
# L2 regularization weight
table_args['lmbda'] = [0.] + [i for i in 10**np.linspace(-4, -1, 4)]
# Initialization of weights
table_args['weight_init'] = ['he_normal', 'he_uniform']

########## Run model ##########

# Collate arguments into input class
cnn_input = ConvnetInputs(filedir, lr, bs, epochs, n_train,
                                 gen_args, table_args=table_args,
                                 save_prefix='./models/run')

#run models
#model.run_models(dir,lr,bs,epochs,n_train,inv_color,rescale,save_models,filter_length,n_filters,lmbda,init)




#==============================================================================
#     parser = argparse.ArgumentParser(description='Keras-based CNN for mapping crater images to density maps.')
#     parser.add_argument("path", 
#     parser.add_argument("Xtrain", type=str, help="Training input npy file")
#     parser.add_argument("Xtest", type=str, help="Testing input npy file")
#     parser.add_argument("Ytrain", type=str, help="Training target npy file")
#     parser.add_argument("Ytest", type=str, help="Testing target npy file")
#     parser.add_argument('--learn_rate', type=float, required=False,
#                         help='Learning rate', default=0.0001)
# #    parser.add_argument('--crater_cutoff', type=int, required=False,
# #                        help='Crater pixel diameter cutoff', default=3)
#     parser.add_argument('--batchsize', type=int, required=False,
#                         help='Crater pixel diameter cutoff', default=32)
#     parser.add_argument('--lambd', type=float, required=False,
#                         help='L2 regularization coefficient', default=0.)
#     parser.add_argument('--epochs', type=int, required=False,
#                         help='Number of training epochs', default=30)
#     parser.add_argument('--f_samp', type=int, required=False,
#                         help='Random fraction of samples to use', default=1.)
#     parser.add_argument('--dumpweights', type=str, required=False,
#                         help='Filename to dump NN weights to file')
#     parser.add_argument('--dumpargs', type=str, required=False,
#                         help='Filename to dump arguments into pickle')
# 
# 
# #    parser.add_argument('--lu_csv_path', metavar='lupath', type=str, required=False,
# #                        help='Path to LU78287 crater csv.', default="./LU78287GT.csv")
# #    parser.add_argument('--alan_csv_path', metavar='lupath', type=str, required=False,
# #                        help='Path to LROC crater csv.', default="./alanalldata.csv")
# #    parser.add_argument('--outhead', metavar='outhead', type=str, required=False,
# #                        help='Filepath and filename prefix of outputs.', default="out/lola")
# #    parser.add_argument('--amt', type=int, default=7500, required=False,
# #                        help='Number of images each thread will make (multiply by number of \
# #                        threads for total number of images produced).')
# 
#     in_args = parser.parse_args()
# 
#     # Print Keras version, just in case
#     print('Keras version: {0}'.format(keras.__version__))
# 
#     # Read in data, normalizing input images
# 
# 
#     # Declare master dictionary of input variables
#     args = {}
# 
#     # Get image and target sizes
#     args["imgshp"] = tuple(Xtrain.shape[1:])
#     args["tgtshp"] = tuple(Ytrain.shape[1:])
# 
#     # Load constants from user
#     args["path"] = args.path
#     args["learn_rate"] = in_args.learning_rate
#     #args["c_pix_cut"] = in_args.crater_cutoff
# 
#     args["batchsize"] = in_args.batchsize
#     args["lambda"] = in_args.lambd
#     args["N_epochs"] = in_args.epochs
#     args["f_samp"] = in_args.f_samp
# 
#     args["dumpweights"] = in_args.dumpweights
# 
#     # Calculate next largest multiple of batchsize to N_train*f_samp
#     # Then use to obtain subset
#     args["N_sub_train"] = int(args["batchsize"] * np.ceil( Xtrain.shape[0] * \
#                                         args["f_samp"] / args["batchsize"] ))
#     args["sub_train"] = np.random.choice(X.shape[0], size=args["N_sub_train"])
#     Xtrain = Xtrain[args["sub_train"]]
#     Ytrain = Ytrain[args["sub_train"]]
# 
# 
#     # Cross-validation parameters that could be user-selectable at a later date.
#     args["CV_lambd_N"] = 10             # Number of L2 regularization coefficients to try
#     args["CV_lambd_range"] = (-3, 0)    # Log10 range of L2 regularization coefficients
#     args["random_state"] = None         # Random initializer for train_test_split
#     args["test_size"] = 0.2             # train_test_split fraction of examples for validation
# 
# 
#     # CALL TRAIN TEST
# 
#     if in_args.dumpargs:
#         pickle.dump( args, open(in_args.dumpargs, 'wb') )
#==============================================================================
