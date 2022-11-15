#!/usr/bin/env python
# coding: utf-8

# # Data Processor - Chaotic Data:
# - Lorenz
# - ...

# ### Auxiliary Initializations

# In[10]:


# # Test-size Ratio
# test_size_ratio = 1-(1/48)
# min_width = 200
# # Ablation Finess
# N_plot_finess = 10
# # min_parts_threshold = .001; max_parts_threshold = 0.9
# N_min_parts = 1; N_max_plots = 10
# Tied_Neurons_Q = True
# # Partition with Inputs (determine parts with domain) or outputs (determine parts with image)
# Partition_using_Inputs = True
# # Cuttoff Level
# gamma = .5
# # Softmax Layer instead of sigmoid
# softmax_layer = True

# #TEMP FOR DEBUGGING
# # Load Packages/Modules
# exec(open('Init_Dump.py').read())
# # Load Hyper-parameter Grid
# exec(open('Grid_Enhanced_Network.py').read())
# # Load Helper Function(s)
# exec(open('Helper_Functions.py').read())


# In[11]:


#!/usr/bin/env python
# coding: utf-8

# # Initializations
# Here we dump the list of all initializations needed to run any code snippet for the NEU.

# ---

# In[ ]:

# (Semi-)Classical Regressor(s)
from scipy.interpolate import interp1d
import statsmodels.api as sm
# import rpy2.robjects as robjects # Work directly from R (since smoothing splines packages is better)
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import ElasticNetCV

# (Semi-)Classical Dimension Reducers
from sklearn.decomposition import PCA


# Grid-Search and CV
from sklearn.model_selection import RandomizedSearchCV, KFold
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.pipeline import Pipeline
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error


# Data Structuring
import numpy as np
import pandas as pd


# Pre-Processing
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from scipy.special import expit

# Regression
from sklearn.linear_model import LinearRegression
from scipy import linalg as scila

# Random Forest & Gradient Boosting (Arch. Construction)
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor

# Random-Seeds
import random

# Rough Signals
# from fbm import FBM

# Tensorflow
import tensorflow as tf
from keras.utils.layer_utils import count_params
import keras as K
import keras.backend as Kb
from keras.layers import *
from keras.models import Model
from keras.models import Sequential
from keras import layers
from keras import utils as np_utils
from tensorflow.keras.initializers import RandomUniform
from tensorflow.keras.constraints import NonNeg
from tensorflow.keras.optimizers import Adam

# Operating-System Related
import os
from pathlib import Path
# import pickle
#from sklearn.externals import joblib

# Timeing
import time
import datetime as DT

# Visualization
import matplotlib
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt
import seaborn as sns
import seaborn as sns; sns.set()

# z_Misc
import math
import warnings













########################
# Make Paths
########################

Path('./outputs/models/').mkdir(parents=True, exist_ok=True)
Path('./outputs/models/Benchmarks/').mkdir(parents=True, exist_ok=True)
Path('./outputs/models/Benchmarks/Invertible_Networks/GLd_Net/').mkdir(parents=True, exist_ok=True)
Path('./outputs/models/Benchmarks/Invertible_Networks/Ed_Net/').mkdir(parents=True, exist_ok=True)
Path('./outputs/models/Benchmarks/Linear_Regression/').mkdir(parents=True, exist_ok=True)
Path('./outputs/models/NAIVE_NEU/').mkdir(parents=True, exist_ok=True)
Path('./outputs/models/NEU/').mkdir(parents=True, exist_ok=True)
Path('./outputs/tables/').mkdir(parents=True, exist_ok=True)
Path('./outputs/results/').mkdir(parents=True, exist_ok=True)
Path('./outputs/plotsANDfigures/').mkdir(parents=True, exist_ok=True)
Path('./inputs/data/').mkdir(parents=True, exist_ok=True)


## Melding Parameters (These are put to meet rapid ICML rebuttle deadline when merging codes)
Train_step_proportion = test_size_ratio


# ### Prepare Data
    
# In[20]:


#------------------------#
# Run External Notebooks #
#------------------------#
if Option_Function == "lorenz":
    #--------------#
    # Prepare Data #
    #--------------#
    # Read Dataset - get input data
    lorenz_input_data = pd.read_csv('inputs/lorenz_system/lorenz_input_data.csv')
    # set index
    lorenz_input_data.set_index('index', drop=True, inplace=True)
    lorenz_input_data.index.names = [None]

    # Remove Missing Data
    lorenz_input_data = lorenz_input_data[lorenz_input_data.isna().any(axis=1)==False]
    
    print('lorenz input data', lorenz_input_data)
    
    # Read Dataset - get output data
    lorenz_output_data = pd.read_csv('inputs/lorenz_system/lorenz_output_data.csv')
    # set index
    lorenz_output_data.set_index('index', drop=True, inplace=True)
    lorenz_output_data.index.names = [None]

    # Remove Missing Data
    lorenz_output_data = lorenz_output_data[lorenz_output_data.isna().any(axis=1)==False]
    

    #-------------#
    # Subset Data #
    #-------------#
    # Get indices
    N_train_step = int(round(lorenz_input_data.shape[0]*Train_step_proportion,0))
    N_test_set = int(lorenz_input_data.shape[0] - round(lorenz_input_data.shape[0]*Train_step_proportion,0))
    # # Get Datasets
    X_train = lorenz_input_data[:N_train_step]
    X_test = lorenz_input_data[-N_test_set:]
    
    print('size training input data:', X_train.shape)
    print('size test input data:', X_test.shape)

    ## Coerce into format used in benchmark model(s)
    data_x = X_train
    data_x_test = X_test
    # Get Targets 
    data_y = lorenz_output_data[:N_train_step]
    data_y_test = lorenz_output_data[-N_test_set:]
    
    print('size training output data:', data_y.shape)
    print('size test output data:', data_y_test.shape)

    # Scale Data
    ###scaler = StandardScaler() # Standardize features by removing the mean and scaling to unit variance.
    ###data_x = scaler.fit_transform(data_x) # Fit to data (computing the mean and std to be used for later scaling), then transform it.
    ###data_x_test = scaler.transform(data_x_test) # Perform standardization by centering and scaling.

    # # Update User
    print('#================================================#')
    print(' Training Datasize: '+str(X_train.shape[0])+' and test datasize: ' + str(X_test.shape[0]) + '.  ')
    print('#================================================#')

    # # Set First Run to Off
    First_run = False

    #-----------#
    # Plot Data #
    #-----------#
    #fig = ks_input_data.plot(figsize=(16, 16))
    print('lorenz_input_data shape:',lorenz_input_data.shape)
    #N_plot_ks = 2048
    #x = (2*np.pi*np.arange(1,N_plot_ks+1)/N_plot_ks)
    #h = 0.01
    #tmax = 9.98
    #step_max = round(tmax/h)
    #step_plt = int(tmax/(998*h))
    #print(step_plt)
    #dt = h
    #tt = 0
    #
    #for step in range(1, step_max):
        #t = step*h
        #if step % step_plt == 0:
            #tt = np.hstack((tt, t))
    #
    #fig = plt.figure(figsize=(9,9))
    #fig, ax = plt.subplots(1,1)
    #X, T = np.meshgrid(x, tt)
    #print('X shape and T shape:', X.shape, T.shape)
    #print(type(X), type(T), type(ks_input_data))
    #X = list(X)
    #T = list(T)
    #ks_input_data_plot = np.matrix(ks_input_data)
    #print(type(ks_input_data_plot))
    #im = ax.pcolormesh(X, T, ks_input_data_plot, cmap='inferno', rasterized=True)
    #fig.colorbar(im)
    #plt.show()

    # SAVE Figure to .eps
    #plt.savefig('./outputs/plotsANDfigures/ks_Data.pdf', format='pdf')
    
# In[21]:


# ICML Coercsion
y_train = np.array(data_y)
y_test = np.array(data_y_test)

print('size training output data:', y_train.shape)
print('size test output data:', y_test.shape)


# ---
