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