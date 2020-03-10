# install python 3.7.0 or see official tensorflow installation page for supported versions
# importing libraries
from keras.preprocessing.image import ImageDataGenerator  # used to make more data (crop, flip etc) from 1 image
from keras.models import Sequential  # creates a prediction model
from keras.layers import Conv2D, MaxPooling2D  # extract the features of a image, reduce the size of the data
from keras.layers import Activation, Dropout, Flatten, Dense  # Activation for the nural net, Dropout to restrict the overfit, Flatten makes a 1D array from a 2D image, Dense to create layers
from keras import backend as K  # tells which channel of a Image comes 1st, RGB
from keras.preprocessing import image  # import images and process them
import numpy as np
import os.path

"""Data Generation setup"""
img_width, img_height = 224, 224

train_data_dir = r'PetImages\train'
validation_data_dir = r'PetImages\validation'
nb_train_samples = 100  # sample size training
nb_validation_samples = 50  # sample size for validation
epochs = 50  #10/50  # how many times same picture will be processed
batch_size = 20  #16/20  # how many picture at the same time will be process

# if RGB comes first in image pixel data or not
if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

"""create a nural network model"""
model = Sequential()  # makes a model stacking features on top of another in linear order
model.add(Conv2D(32, (2, 2), input_shape = input_shape))  # convolution nural network extracts 32 features in 3x3 windows at once
model.add(Activation('relu'))  # relu activation function used, output: 1/0
model.add(MaxPooling2D(pool_size =(2, 2)))  # combile 2x2 pixels in 1 and reduce the image feature size

model.summary()  # shows a summary of the model till now

# doing again  the 3 steps
model.add(Conv2D(32, (2, 2)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size =(2, 2)))

# doing again  but with 64 features
model.add(Conv2D(64, (2, 2)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size =(2, 2)))
#TODO: need to repeat these steps if need more accuracy

model.add(Flatten())  # makes 1D array from 2D data
model.add(Dense(64))  # used all the features of the last step
model.add(Activation('relu'))
model.add(Dropout(0.5))  # restricts the model from over fitting
model.add(Dense(1))  # ultimatly making 1 node out of all the 64 nodes
model.add(Activation('sigmoid'))  # sigmoid output is 0-1

model.summary()

# compiles the model data
model.compile(loss ='binary_crossentropy',
                optimizer ='rmsprop',
                metrics =['accuracy'])

"""Data Generation"""
# Making the Training set. Makes 4 different images from 1 image
train_datagen = ImageDataGenerator(
                rescale = 1.0 / 255,
                shear_range = 0.2,
                zoom_range = 0.2,
                horizontal_flip = True)

# Making test data set
test_datagen = ImageDataGenerator(rescale = 1.0 / 255)

# generating training data from all the images
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size =(img_width, img_height),
    batch_size = batch_size,
    class_mode ='binary')

# generating validation data from all the images
validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size =(img_width, img_height),
    batch_size = batch_size,
    class_mode ='binary')

# fitting the model
model.fit_generator(train_generator,
    steps_per_epoch = nb_train_samples // batch_size,
    epochs = epochs, validation_data = validation_generator,
    validation_steps = nb_validation_samples // batch_size)

model.save_weights('model_saved.h5')  # saves model data, next time HW will use.

# prepare the target image for process
# img_pred  = image.load_img(r'testModel/Dog/11458.jpg', target_size = (224, 224))
img_pred  = image.load_img(r'testModel/Cat/8909.jpg', target_size = (224, 224))
img_pred = image.img_to_array(img_pred)
img_pred = np.expand_dims(img_pred, axis = 0)  # converts into numpy array

"""test the model and see the prediction"""
rslt = model.predict(img_pred)
print(rslt)
if rslt[0][0] == 1:
    prediction = 'Dog'
else:
    prediction = 'Cat'

print('=== {} ==='.format(prediction))