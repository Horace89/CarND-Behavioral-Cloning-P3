
# coding: utf-8
# # P3 - Behavioral Cloning


import os.path
import csv
import glob
import json
import random
import time

import numpy as np

import cv2

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from keras.optimizers import Adam, RMSprop
from keras.layers.core import Dense, Activation
from keras.layers import Dense, Activation, Dropout,Conv2D,Convolution2D,MaxPooling2D,Flatten,Lambda
from keras.models import Sequential, Model
from keras import backend as K
from keras.regularizers import l2
from keras import callbacks
from keras.models import model_from_json

import pandas as pd


from keras import __version__ as keras_version
print('Keras version:', keras_version)






# ## DATA EXPLORATION
#
# - all images are stored in data/IMG/
# - `driving_log.csv` contains the paths of acquired images (for training) and the corresponding parameter values:
#
#     *center_camera, left_camera, right_camera, steering_angle, throttle, brake, speed *


DATA_PATH, has_header = 'data/', True

training_dat = pd.read_csv(DATA_PATH + 'driving_log.csv', names=None)



with open(DATA_PATH + 'driving_log.csv', 'r') as csvfile:
    file_reader = csv.reader(csvfile, delimiter=',')
    log = list(file_reader)

log = np.array(log)
if has_header: log = np.delete(log, (0), axis=0) #remove the header

print('Dataset:\n {} images | Number of steering data: {}'.format(3 * len(log), len(log)))



# Sanity check: the number of images listed in 'driving_log.csv' and the ones present in 'data/IMG' are equal.
ls_imgs = glob.glob(DATA_PATH + 'IMG/*.jpg')
print('len(ls_imgs):', len(ls_imgs))
assert len(ls_imgs) == len(log) * 3, 'Actual number of *jpg images does not match with the csv log file'


# ## Image Visualization
# The images are RGB format.

# In[7]:


#index of columns in: center_camera, left_camera, right_camera, steering_angle, throttle, brake, speed
CENTER_CAM, LEFT_CAM, RIGHT_CAM, STEERING_ANGLE = range(4)

center_im = log[:, CENTER_CAM]
left_im = log[:, LEFT_CAM]
right_im = log[:, RIGHT_CAM]
steering_rad = log[:, STEERING_ANGLE].astype(float)



steering_deg = np.rad2deg(steering_rad).astype(int)


unique_st_angles, counts = np.unique(steering_deg, return_counts=True)
# get unique values
print('Number of unique angles value : {}'.format(len(unique_st_angles)) )



cnt_neg = np.sum(steering_deg < 0)
cnt_zero = np.sum(steering_deg == 0)
cnt_pos = np.sum(steering_deg > 0)
print('Count of positive angles: {} | zero angle: {} | negative angles: {}'.format(cnt_pos, cnt_zero, cnt_neg))




# In[21]:


CROPPED_DIM = 64

def random_crop(image, steering=0.0, tx_min=-20, tx_max=20, ty_min=-2, ty_max=2, rand=True):
    """
    We will randomly crop subsections at the approximate center of the image and use them as our data set.
    After that we downside the resulting image to 64x64.
    """

    col_start, col_end = abs(tx_min), image.shape[1] - tx_max
    horizon, bonnet = 60, 136

    if rand:
        tx = np.random.randint(tx_min, tx_max+1)
        ty = np.random.randint(ty_min, ty_max+1)
    else:
        tx, ty = 0, 0

    #cropping
    random_crop = image[horizon+ty:bonnet+ty, col_start+tx:col_end+tx, :]

    #downsizing
    image = cv2.resize(random_crop,(CROPPED_DIM,CROPPED_DIM), cv2.INTER_AREA)

    # the steering variable needs to be updated to counteract the shift
    delta_steering = -tx/(tx_max - tx_min)/3.0 if tx_min != tx_max else 0
    steering += delta_steering

    return image, steering


def random_shear(image, steering, shear_range=40):
    rows, cols, ch_ = image.shape

    dx = np.random.randint(-shear_range, shear_range + 1)

    random_point = [cols//2+ dx, rows//2]
    pts1 = np.float32([[0,rows], [cols,rows], [cols//2,rows//2]])
    pts2 = np.float32([[0,rows], [cols,rows], random_point])

    M = cv2.getAffineTransform(pts1,pts2)
    image = cv2.warpAffine(image,M,(cols,rows),borderMode=1)

    delta_steering = dx/(rows/2) * 360/(2*np.pi*25.0) / 6.0
    steering += delta_steering

    return image, steering


def random_brightness(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    alpha = np.random.uniform(low=.2, high=.9, size=None)
    hsv[:,:,2] = (hsv[:,:,2] * alpha).astype('uint8')
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    return rgb


def random_flip(image,steering):
    coin=np.random.randint(0, 2)
    if coin == 0:
        image = cv2.flip(image,1)
        steering = -steering

    return image, steering



def read_image_and_steering(idx, cam_choice, left_im_, center_im_, right_im_, steer):
    """
    idx: index of image
    cam_choice: center_im / left_im / right_im
    """
    assert cam_choice in 'LCR'
    offset = 1.2
    dist = 100.0

    cam = {'C':center_im_, 'L':left_im_, 'R':right_im_}[cam_choice]
    image = cv2.imread(DATA_PATH + cam[idx].strip()) #beware of leading whitespaces!
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if cam_choice == 'C':
        delta_steering = 0
    elif cam_choice == 'L':
        delta_steering = -offset/dist * 360/(2*np.pi)/25.0
    elif cam_choice == 'R':
        delta_steering = offset/dist * 360/(2*np.pi)/25.0

    steering = steer[idx] + delta_steering

    return image, steering




X_train = training_dat[['left','center','right']]
Y_train = training_dat['steering']

X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.1, random_state=42)


# In[25]:


# we convert the input data into Numpy arrays
X_left  = X_train['left'].as_matrix()
X_right = X_train['right'].as_matrix()
X_train = X_train['center'].as_matrix()

Y_train = Y_train.as_matrix().astype(np.float32) #we don't need float64

#for validation, we only keep the center image
X_val = X_val['center'].as_matrix()

Y_val = Y_val.as_matrix().astype(np.float32) #we don't need float64



def generate_training_example(left_im_, center_im_, right_im_, steer, drop_small_angle_prob = 0.95, verbose=False):
    """
    drop_small_angle_prob =  2 : we drop all examples with a 0 angle
    drop_small_angle_prob = -1 : we keep all examples with a 0 angle
    """

    ONE_DEGREE_THRESHOLD = 0.0175 #one degree in radian
    while True:
        idx = np.random.randint(0, len(steer))
        if np.absolute(steer[idx]) > ONE_DEGREE_THRESHOLD or             np.random.uniform(0,1) > drop_small_angle_prob: #we keep only very few training examples with a ~0 steering angle
            break

    if verbose:
        print('training example at index: {}'.format(idx))

    lcr = random.choice('LCR')
    if verbose:
        print('L/C/R: {}'.format(lcr))

    image, steering = read_image_and_steering(idx, lcr, left_im_, center_im_, right_im_, steer)
    if verbose:
        print('steering: {}'.format(steering))
        plt.imshow(image)

    image, steering = random_shear(image, steering, shear_range=40)
    if verbose:
        print('steering: {}'.format(steering))
        plt.figure()
        plt.imshow(image)

    image, steering = random_crop(image, steering)
    if verbose:
        print('steering: {}'.format(steering))
        plt.figure()
        plt.imshow(image)

    image, steering = random_flip(image,steering)
    if verbose:
        print('steering: {}'.format(steering))
        plt.figure()
        plt.imshow(image)

    image = random_brightness(image)
    if verbose:
        plt.figure()
        plt.imshow(image)

    return image.astype('uint8'), steering.astype('float32')



def get_validation_set(X_val, Y_val):

    images = np.empty((len(X_val), CROPPED_DIM, CROPPED_DIM,3), dtype='uint8')
    steerings = np.empty(len(X_val), dtype='float32')
    for i in range(len(X_val)):
        #quick hack: we put 'None' as the left and right cam images for the validation set,
        #and we always chose 'C' for the center camera.
        image, steering = read_image_and_steering(i, 'C', None, X_val, None, Y_val)

        #hack: we do not crop images for the validation set: all cropping params are set to 0.
        #TODO : set 'rand=False' param
        images[i], steerings[i] = random_crop(image, steering, tx_min=0, tx_max=0, ty_min=0, ty_max=0)

    return images, steerings



def generate_training_batch(left_im_, center_im_, right_im_, steer, batch_size = 32):

    batch_images = np.empty((batch_size, CROPPED_DIM, CROPPED_DIM, 3), dtype='uint8')
    batch_steerings = np.empty(batch_size, dtype='float32')
    while True:
        for i in range(batch_size):
            image, steering = generate_training_example(left_im_, center_im_, right_im_, steer)
            batch_images[i] = image
            batch_steerings[i] = steering

        yield batch_images, batch_steerings


# # CNN model

model = Sequential()
model.add(Lambda(lambda x: x/127.5 - 1.0,input_shape=(CROPPED_DIM,CROPPED_DIM,3)))
model.add(Convolution2D(32, 8,8 ,border_mode='same', subsample=(4,4)))
model.add(Activation('relu'))
model.add(Convolution2D(64, 8,8 ,border_mode='same',subsample=(4,4)))
model.add(Activation('relu',name='relu2'))
model.add(Convolution2D(128, 4,4, border_mode='same',subsample=(2,2)))
model.add(Activation('relu'))
model.add(Convolution2D(128, 2,2, border_mode='same',subsample=(1,1)))
model.add(Activation('relu'))
model.add(Flatten())

model.add(Dropout(0.5))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(128))
model.add(Dense(1))

model.summary()



BATCH_SIZE = 256

train_generator = generate_training_batch(X_left, X_train, X_right, Y_train, batch_size=BATCH_SIZE)

X_val, Y_val = get_validation_set(X_val,Y_val)


# In[33]:


# ## Let's start the training!

adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(optimizer=adam, loss='mse')

NB_EPOCH = 3

t0 = time.time()

history_object = model.fit_generator(train_generator,
                    samples_per_epoch= 20000, #len(Y_train),
                    nb_epoch=NB_EPOCH,
                    validation_data=(X_val,Y_val),
                    verbose=1)

print("\nTraining time:", time.time() - t0)


# In[61]:


model.save('model.h5')

# launch sim with `python drive.py model.h5`
