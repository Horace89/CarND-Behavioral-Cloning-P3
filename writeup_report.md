# **Behavioral Cloning**


---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report



## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

In this submission, only the original Udacity training data was used (which may be downloaded at https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip)

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network.
This file is a copy of the code that is commented in the `model.ipynb` Jupyter notebook:
- this notebook demonstrates my exploration of data augmentation tactics.
- it shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.


### Model Architecture and Training Strategy


#### 1. Context

The simulated car is equipped with three cameras (left/center/right) which provide images from three different view points. The training track has sharp corners, exits, entries, bridges, partially missing lane lines and changing light conditions. The model developed here was trained exclusively on the training track and completes it.

The main problem lies in the bias of the original Udacity training data: many pictures correspond to a steering angle of zero. The exploration of the dataset may be found in the `model.ipynb` notebook.

The left-right skew is less problematic and can be eliminated by flipping images and steering angles at he same time. However, even after balancing left and right angles most of the time the steering angle during normal driving is small or zero and thus introduces a bias towards driving straight.

If we only train on this unmodified dataset, the car leaves the track quickly. As described in the lessons, one way to balance this problem would be to purposely let the car drift towards the side of the road and to start recovery in the lasts moments. However, I did not use this approach and only used data augmentation in order to generate enough appropriate data for the training.


#### 2. My strategy: data augmentation

CNNs architectures have been successfully used to predict the steering angle of the simulator. In [this article](https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9#.4iywd3mzj)
 Vivek Yadav described a solution to this steering problem based on using data augmentation. My submission draws on the insights obtained there and on ideas from the forum and the Slack channel: I used many augmentation techniques to enrich the set of training pictures so that:
- it contains more pictures
- we get more pictures with sharp steering angles




### Model Architecture and Training Strategy

#### 1. Solution Design Approach

In this project, a single variable (the current steering angle) is predicted as a real number. The problem is thus a regression task.

My initial approach was to use [LeNet](http://yann.lecun.com/exdb/lenet/), but it was hard to keep the car inside the track. I decided to use more architecture versions of CNNs. I did not want to use the [nVidia](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/) model, as many students have followed this path, and I wanted to experiment a bit on what could work from scratch.

First of all, I implemented a preprocessing step which downsizes every image to 64x64 pixels (see `random_crop` function).

Note: the `drive.py` file had to be modified to include this preprocessing too.

Each image gets normalized to the range [-1,1] otherwise no preprocessing is performed. Following the input layer are 4 convolutional layers. ReLU activations are used.
- the first two convolutional layers employ kernels of size k=(8,8) with a stride of s=(4,4) and 32 and 64 channels, respectively.
- the next convolutional layer uses k=(4,4) kernels, a stride of s=(2,2) and 128 channels. In the last convolutional layer we use k=(2,2), a stride s=(1,1) and again 128 channels.

Following the convolutional layers are two fully connected layers with ReLU activations as well as dropout regularization before the layers. A dropout factor of 0.5 proved to work.

The final layer is a single neuron that provides the predicted steering angle.



#### 2. Creation of the Training Set & Training Process


All the training was done on AWS and on my iMac laptop. I also have bought a server with a Nvidia 1070 GPU and I will continue to work on this project on this new machine.

After reading a few blog posts and exploring the original training data set, I decided early on that I wanted to try to get a working model without any addes recorded training data, but instead only using an augmentation strategy.

I simulated variations in driving events by transforming (shifts, shears, crops, brightness, flips) the set of recorded images (using OpenCV) with corresponding steering angle changes. The final training images are then generated on the fly with Python generators (see the `generate_training_example` function) in batches of 256 with 20000 images per epoch.


The performed operations are (see the `generate_training_example` function):

- A random training example is chosen
- if its steering angle is more than 1 degree we keep it. If not (ie the angle is ~0), we throw a dice and keep this example only in 10% of the cases
- the camera is chosen randomly between left, right, center
- random shear: the image is sheared horizontally to simulate a bending road
- random crop: we randomly crop a frame out of the image to simulate the car being offset from the middle of the road (also downsizing the image to 64x64x3 is done in this step)
- random horizontal flipping: we flip the image left-right with a 50-50 probability
- random brightness: to simulate different lighting conditions

In steps left&right camera/random-shear/crop/horizontal flipping, the steering angle is adjusted to account for the change in the image. The cascade of these transforming operations leads to a practically infinite number of different examples that could be used in a training phase.


#### 3. Validation data and optimizer

For validation purposes 10% of the training data (about 1000 images) was held back (see the line `X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.1, random_state=42)`).

Only the center camera images are used for validation. After few epochs (~3) the validation loss becomes stable. The validation loss stays at 1/3 of the training loss, which is certainly due to the fact that validation data and training data do not come from the same distribution, as validation data has not been augmented. Even more important: validation data has many examples with ~0 steering angles, which is not the case for training data as these examples were pruned aggressively (90% chance, see above section).

I used the Adam optimizer for training. All training was performed with the fastest graphics setting in the simulator.


### Conclusion

By making a large use of image augmentation techniques, I have been able to train a neural network to recover the car from rare events, like suddenly appearing curves. This model is able to drive the car on track 1.
