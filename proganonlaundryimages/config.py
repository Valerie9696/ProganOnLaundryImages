from __future__ import division
import os
import time
import math
from functools import partial
import numpy as np
import matplotlib.pyplot as plt

from IPython.display import clear_output

import tensorflow as tf
from tensorflow.keras.layers import Layer, InputSpec, Conv2D, Conv1D, Conv2DTranspose, Activation, Reshape, LayerNormalization, BatchNormalization, UpSampling2D
from tensorflow.keras.layers import Input, UpSampling2D, Dropout, Concatenate, Add, Dense, Multiply, LeakyReLU, Flatten, AveragePooling2D, Multiply
from tensorflow.keras import initializers, regularizers, constraints, Model, Sequential
from tensorflow.keras import backend as K
from tensorflow.keras.utils import Sequence
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.utils import to_categorical, plot_model

# Compute on CPU
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
tf.config.list_physical_devices('GPU')
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
AUTOTUNE = tf.data.experimental.AUTOTUNE

tf.config.experimental_run_functions_eagerly(True)

MODEL_NAME = 'PGGAN'
DATA_BASE_DIR = './dataset' # Modify this to your dataset path.
SEGMENTATION_DIR = './segmentation'
OUTPUT_PATH = 'outputs'
MODEL_PATH = 'models'
TRAIN_LOGDIR = os.path.join("logs", "tensorflow", 'train_data') # Sets up a log directory.
if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)
    
batch_size = 16
# Start from 4 * 4
image_size = 3
image_width = 5
NOISE_DIM = 512
LAMBDA = 10

EPOCHs = 160
CURRENT_EPOCH = 1 # Epoch start from 1. If resume training, set this to the previous model saving epoch.

total_data_number = len(os.listdir(DATA_BASE_DIR))

# To reduce the training time, this number is lower than the original paper,
# thus the output quality would be worse than the original.
switch_res_every_n_epoch = 20
#switch_res_every_n_epoch = math.ceil(800000 / total_data_number)

SAVE_EVERY_N_EPOCH = 5 # Save checkpoint at every n epoch

LR = 1e-3
BETA_1 = 0.
BETA_2 = 0.99
EPSILON = 1e-8
# Decay learning rate
MIN_LR = 0.000001
DECAY_FACTOR=1.00004

# profiler settings
PROF_OPTIONS = tf.profiler.experimental.ProfilerOptions(
    host_tracer_level=2,
    python_tracer_level=0,
    device_tracer_level=1
)
# set True to measure hardware resource use
PROFILE = False

# Creates a file writer for the log directory.
file_writer = tf.summary.create_file_writer(TRAIN_LOGDIR)
#test_file_writer = tf.summary.create_file_writer(TEST_LOGDIR)

if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)
