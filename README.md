# Lunar crater counting using Convolutional Networks

This repo contains the essential pieces of code to train a convnet to recognize craters on the moon. 

# Getting the Data
Create a folder on your home directory to house all your data. By default this repo contains the folder 'dataset' for this purpose. This will correspond to the ‘dir’ variable in run_moon_convnet_model.py

From /scratch/r/rein/silburt/ on scinet, get the following numpy files, and copy them to the following folders within the ‘dir’ directory:

*File*	->			*Folder within ‘dir’*  
train_data.npy   	->	Train_rings/  
train_target.npy   	-> 	Train_rings/  
dev_data.npy		->	Dev_rings/  
dev_target.npy		->	Dev_rings/  
test_data.npy		->	Test_rings/  
test_target.npy		->	Test_rings/  
custom_loss_csvs.npy	->	Dev_rings_for_loss/  
custom_loss_images.npy	->	Dev_rings_for_loss/  

# Running the Code
All the code is contained within run_moon_convnet_model.py. Before running the code one must first load the following modules/virtual environments:

module load gcc/6.2.1  
module load cuda/8.0  
source /home/k/kristen/kristen/keras_venv_P8.v2/bin/activate  

To execute the code, use the following kind of command: CUDA_VISIBLE_DEVICES=2 nohup python run_moon_convnet_model.py > output.txt &  
First, make sure that the chosen CUDA device is available by doing 'nvidia-smi'.

Within run_moon_convnet_model.py, all the main parameters that you might want to change is at the bottom of the script, under the __main__ function. These parameters have explanations given. For iterating over parameters (i.e. performing a grid search), in __main__ look for:  
########## Parameters to Iterate Over ##########  
I’ve given a simple example of how to do this in the code. These variables must always be lists, even if you only want to run one model. If *save_models=1*, a model will be saved for every set of parameters you iterate over. Look for *model.save()* within the *train_and_test_model()* function and make sure that the name assigned to each model is unique so that models wont get overwritten as you iterate.

# Generating/Analyzing model predictions
Once you have successfully created and trained a model, you will want to analyze new predictions. This can be done using rings_analyze_remote.py, which allows you to generate model predictions on scinet, and then analyze them on your local system.  
Step 1 - On scinet, place the desired models in the models/ folder (by default models generated by run_moon_convnet_model.py are placed here). Add the name of the model to the 'models' array in the argument list at the bottom of rings_analyze_remote.py.  
Step 2 - Set the other parameters and run the script rings_analyze_remote.py on scinet.  
Step 3 - Copy the generated predictions (in the form of numpy arrays) to your local machine.  
Step 4 - Open up rings_analyze_remote.ipynb and execute the cells to analyze the predictions locally.  
