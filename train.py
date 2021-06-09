import datetime
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#define the paths
training_path = '/content/drive/MyDrive/Colab Notebooks/learnbay_deep_learning/dataset/cat_dog_data/training_set'
testing_path = '/content/drive/MyDrive/Colab Notebooks/learnbay_deep_learning/dataset/cat_dog_data/test_set'
prediction_image_path = '/content/drive/MyDrive/Colab Notebooks/learnbay_deep_learning/dataset/cat_dog_data/single_prediction'


train_datagen = ImageDataGenerator(rescale = 1./255, #All the pixel values would be 0-1
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)


# preparing the data
training_set = train_datagen.flow_from_directory(training_path,
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory(testing_path,
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')


# lets see the data
training_data = next(training_set)
label_data = next(test_set) 


# define the architecture

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
#input_shape goes reverse if it is theano backend
#Images are 2D
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))

# Step 2 - Pooling
#Most of the time it's (2,2) not loosing many. 
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
#Inputs are the pooled feature maps of the previous layer
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
#relu - rectifier activation function
#128 nodes in the hidden layer
classifier.add(Dense(units = 128, activation = 'relu')) 
#Sigmoid is used because this is a binary classification. For multiclass softmax
classifier.add(Dense(units = 1, activation = 'sigmoid'))

# Compiling the CNN
#adam is for stochastic gradient descent 
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


classifier.fit(training_set,
                         epochs = 10,
                         validation_data = test_set)


#Saing the model as a Hierarchical Data Format 5 file 
classifier.save('model/classifier_81.h5')