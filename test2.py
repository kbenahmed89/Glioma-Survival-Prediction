import numpy as np
np.random.seed(1234)
import os
from sklearn.model_selection import StratifiedKFold
from keras.models import load_model
import tensorflow as tf
from tensorflow.python.framework.ops import reset_default_graph
#from ConfusionMatrix import ConfusionMatrix
import SimpleITK as sitk
from scipy import ndimage
import glob, os
import numpy as np
from nibabel.testing import data_path
import nibabel as nib
#import matplotlib.pyplot as plt
from nilearn import datasets
from nilearn import image
from nilearn.plotting import plot_anat, show
import csv
from skimage.transform import downscale_local_mean
from imutils import paths
from nilearn.input_data import NiftiMasker
import matplotlib.pyplot as plt
from nilearn import datasets
from nilearn import image
from nilearn.plotting import plot_anat, show
from nilearn.plotting import plot_roi
from nilearn.image.image import mean_img
import tensorlayer as tl
import os, csv, random, gc, pickle
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
import re
from keras.preprocessing.image import ImageDataGenerator
from utils.nifti_image import NIfTIImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras.optimizers import RMSprop
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras import backend as K
from keras.layers import LeakyReLU
from keras.layers import GlobalAveragePooling3D
from keras.layers.convolutional import Convolution3D, MaxPooling3D, ZeroPadding3D
from keras.utils import multi_gpu_model

    ###======================== HYPER-PARAMETERS ============================###
batch_size = 1
lr = 0.001 
lr_decay = 0.5
decay_every = 100
beta1 = 0.9
n_epoch = 100
print_freq_step = 100

data_types_mean_std_dict =  {'mean': 0.0, 'std': 1.0}
IMAGE_DIMS = (160, 160, 110,1)
test_dir_1='/data/cnn-keras-3D-allmodalities/brats_testCropped_3modalities/T1C'
test_dir_2='/data/cnn-keras-3D-allmodalities/brats_testCropped_3modalities/F'
test_dir_3='/data/cnn-keras-3D-allmodalities/brats_testCropped_3modalities/T2'
num_niftifiles=46
G=1

params = {'target_size': (IMAGE_DIMS[0], IMAGE_DIMS[1],IMAGE_DIMS[2]),
        'batch_size': batch_size,
        'class_mode': 'binary',
	'shuffle':False
         
    }
print("[INFO] loading data...")

train_datagen = NIfTIImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True)

def generate_generator_multiple(generator,dir1, dir2, dir3, params):

    genX1 = generator.flow_from_directory(dir1,**params, seed=1)
    genX2 = generator.flow_from_directory(dir2,**params, seed=1)
    genX3 = generator.flow_from_directory(dir3,**params, seed=1)
    for x in range(1,47,1):
            X1i = genX1.next()
            X2i = genX2.next()
            X3i = genX3.next()
            yield [X1i[0], X2i[0],X3i[0]], X3i[1]  #Yield both images and their mutual label

            


    

import xlwt 
from xlwt import Workbook 
  
# Workbook is created 
wb = Workbook() 
  
# add_sheet is used to create sheet. 
sheet1 = wb.add_sheet('Sheet 1') 
  
import gc  

print("[INFO] loading network...")
row=0
for epo in range(1,11,1):
    column=0    
    print("epoch:"+str(epo) )
    model = load_model("cosinesnapshot_model_"+str(epo)+".h5")    

    inputgenerator=generate_generator_multiple(generator=train_datagen,
                                           dir1=test_dir_1,
                                           dir2=test_dir_2,
                                           dir3=test_dir_3, params=params)      
    scores=[]
    scores = model.predict_generator(inputgenerator,46)
    #print(scores)
    imagePaths = sorted(list(paths.list_files("brats_testCropped_3modalities/T1C")))
    correct = 0
    for i, n in enumerate(imagePaths):
            sheet1.write(row, column, str(scores[i][0]) )
            #sheet1.write(row, column, n )
            column=column+1
    row=row+1
    K.clear_session()
    gc.collect()
    del model
wb.save('results.xls')
