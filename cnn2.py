"""A file containing the convolutional neural network for training over a set of cat and dog pictures."""

from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator

# initialize the CNN
classifier = Sequential()

# the convolution - taking a picture, passing through the feature detectors and obtaining feature maps
classifier.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation="relu"))

# carrying out the max pooling
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# flattening the input pooled data. flattens the pooled feature maps and puts them into a single vector
classifier.add(Flatten())

# we have now reached the full-connection step.
classifier.add(Dense(units=128, activation="relu"))
classifier.add(Dense(units=1, activation="sigmoid"))  # because binary outcome, if not binary use softmax

# compiling the CNN
classifier.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# The data is to be augmented and passed through the created CNN

train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size=(64, 64),  # enter the dimensions expected by the CNN
                                                 batch_size=32,
                                                 class_mode='binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size=(64, 64),
                                            batch_size=32,
                                            class_mode='binary')

classifier.fit_generator(training_set,
                         steps_per_epoch=8000,  # number of images chosen for the training sets
                         epochs=25,
                         validation_data=test_set,
                         validation_steps=2000)  # number of images chosen for the test sets
