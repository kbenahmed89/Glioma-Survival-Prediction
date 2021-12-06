import keras
from math import pi
from math import cos
from math import floor
from keras import backend
import numpy as np
np.random.seed(1234)
import timeit
import os
import tensorflow as tf
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
from nilearn.input_data import NiftiMasker
import matplotlib
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
from keras.layers.convolutional import Conv2D,Conv3D
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
from keras.optimizers import SGD
from keras.layers import Input,concatenate
from keras.models import Model
from keras.layers.convolutional import Conv3D

    ###======================== HYPER-PARAMETERS ============================###
batch_size = 1
lr = 0.0001
lr_decay = 0.5
decay_every = 100
beta1 = 0.9
n_epoch = 500
print_freq_step = 100
data_types_mean_std_dict =  {'mean': 0.0, 'std': 1.0}
IMAGE_DIMS = (160, 160, 110,1)
train_dir_1='/data/cnn-keras-3D-allmodalities/train_3modalities/T1C'
train_dir_2='/data/cnn-keras-3D-allmodalities/train_3modalities/F'
train_dir_3='/data/cnn-keras-3D-allmodalities/train_3modalities/T2'
G=1
num_niftifiles=9291
num_val=12

params = {'target_size': (IMAGE_DIMS[0], IMAGE_DIMS[1],IMAGE_DIMS[2]),
        'batch_size': batch_size,
        'class_mode': 'binary'
    }

train_datagen = NIfTIImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True)

def generate_generator_multiple(generator,dir1, dir2, dir3, params):

    genX1 = generator.flow_from_directory(dir1,**params,seed=1)
    
    genX2 = generator.flow_from_directory(dir2,**params,seed=1)

    genX3 = generator.flow_from_directory(dir3,**params,seed=1)

    while True:
            X1i = genX1.next()
            X2i = genX2.next()
            X3i = genX3.next()
            yield [X1i[0], X2i[0],X3i[0]], X3i[1]  #Yield both images and their mutual label
            
            
inputgenerator=generate_generator_multiple(generator=train_datagen,
                                           dir1=train_dir_1,
                                           dir2=train_dir_2,
                                           dir3=train_dir_3, params=params)

inputShape = (IMAGE_DIMS[0], IMAGE_DIMS[1],IMAGE_DIMS[2],IMAGE_DIMS[3])

def create_convolution_layers(input_img):

     model = Conv3D(20, (3, 3, 3), activation='relu',
                            padding='same', strides=(1, 1, 1),input_shape=inputShape) (input_img)
     model = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                           padding='valid')(model)
     model = Conv3D(40, (3, 3, 3), activation='relu',
                            padding='same', strides=(1, 1, 1)) (model)
     model = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                           padding='valid')(model)
     return model

T1_input = Input(shape=inputShape)
T1_model = create_convolution_layers(T1_input)

FLAIR_input = Input(shape=inputShape)
FLAIR_model = create_convolution_layers(FLAIR_input)

T2_input = Input(shape=inputShape)
T2_model = create_convolution_layers(T2_input)

conv = concatenate([T1_model, FLAIR_model,T2_model])

conv= GlobalAveragePooling3D()(conv)
dense = Dense(64, activation='relu')(conv)
dense = Dropout(0.5)(dense)

output = Dense(1, activation='sigmoid')(dense)

model = Model(inputs=[T1_input, FLAIR_input,T2_input], outputs=[output])

print(model.summary(line_length=150))

############### CALLBACKS ###############
filepath="multibranch-model-improvement-{epoch:02d}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='train_acc', verbose=1)
#callbacks_list = [checkpoint]
############### TRAINING ###############



print("[INFO] compiling model...")
#opt = SGD(lr=lr,decay=lr/n_epoch, momentum=0.9, nesterov=True)
opt = SGD(momentum=0.9)
model.compile(loss="binary_crossentropy", optimizer=opt,metrics=["accuracy"])
class SnapshotEnsemble(keras.callbacks.Callback):
	# constructor
	def __init__(self, n_epochs, n_cycles, lrate_max, verbose=0):
		self.epochs = n_epochs
		self.cycles = n_cycles
		self.lr_max = lrate_max
		self.lrates = list()
 
	# calculate learning rate for epoch
	def cosine_annealing(self, epoch, n_epochs, n_cycles, lrate_max):
		epochs_per_cycle = floor(n_epochs/n_cycles)
		cos_inner = (pi * (epoch % epochs_per_cycle)) / (epochs_per_cycle)
		return lrate_max/2 * (cos(cos_inner) + 1)
 
	# calculate and set learning rate at the start of the epoch
	def on_epoch_begin(self, epoch, logs={}):
		# calculate learning rate
		lr = self.cosine_annealing(epoch, self.epochs, self.cycles, self.lr_max)
		# set learning rate
		backend.set_value(self.model.optimizer.lr, lr)
		# log value
		self.lrates.append(lr)
 
	# save models at the end of each cycle
	def on_epoch_end(self, epoch, logs={}):
		# check if we can save model
		epochs_per_cycle = floor(self.epochs / self.cycles)
		if epoch != 0 and (epoch + 1) % epochs_per_cycle == 0:
			# save model to file
			filename = "cosinesnapshot_model_%d.h5" % int((epoch + 1) / epochs_per_cycle)
			self.model.save(filename)
			print('>saved snapshot %s, epoch %d' % (filename, epoch))

n_epochs = 500
n_cycles = n_epochs / 50
ca = SnapshotEnsemble(n_epochs, n_cycles, 0.0001)
callbacks_list =[ca]
print("[INFO] training network...")
H = model.fit(inputgenerator,
                        steps_per_epoch=num_niftifiles// (batch_size * G ),  # total number of images
                        epochs=n_epoch,
                        #validation_data=valid_generator,validation_steps= num_val // (batch_size * G ),
                        callbacks=callbacks_list, verbose=2)

history_dict = H.history
print(history_dict.keys())

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
N = n_epoch
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="upper left")
plt.savefig("multibranch_cosine")

