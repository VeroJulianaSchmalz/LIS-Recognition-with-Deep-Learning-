### Import necessary modules 

import tensorflow as tf 

from tensorflow.keras.models                import Sequential, Model
from tensorflow.keras.layers                import *
from tensorflow.keras.preprocessing.image   import ImageDataGenerator
from tensorflow.keras.utils                 import to_categorical
from tensorflow.keras.optimizers            import SGD, RMSprop, Adam, Adagrad, Adadelta

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.utils import class_weight

import matplotlib.pyplot as plt
import random
import cv2
import pandas as pd
import numpy as np
import matplotlib.gridspec as gridspec
import seaborn as sns
import sklearn
import scipy
from skimage.transform import resize
import csv
from tqdm import tqdm
from sklearn import model_selection
from sklearn.model_selection import train_test_split, learning_curve,KFold,cross_val_score,StratifiedKFold
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

### Data augmentation 

#Definition of variables 
bs = 64    
k = 2


#features to generate a dataset of images given the following same characteristics

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=(0.3),
        zoom_range=(0.3),
        width_shift_range=(0.2),
        height_shift_range=(0.2),
        brightness_range=(0.05,0.85),
        horizontal_flip=False)


# Loading the images from the following directories of TRAIN and VALIDATION 

train_generator = train_datagen.flow_from_directory(
        'folder/to_training_dataset/with_letters_subfolders',
        class_mode='categorical',
        shuffle=True,
        target_size=(28*k, 28*k),
        color_mode = 'rgb', 
        subset = 'training',
        batch_size=bs)

valid_generator = train_datagen.flow_from_directory(
        'folder/to_validation_dataset/with_letters_subfolders',
        class_mode='categorical',
        shuffle=True,
        target_size=(28*k, 28*k),
        color_mode = 'rgb', 
        batch_size=bs)

test_generator = train_datagen.flow_from_directory(
        'folder/to_test_dataset/with_letters_subfolders',
        class_mode='categorical',
        shuffle=False,
        target_size=(28*k, 28*k),
        color_mode = 'rgb', 
        batch_size=bs)

# get classes' indices 

print(train_generator.class_indices)

### Import pre-trained model from https://arxiv.org/abs/1409.1556

import tensorflow as tf    
model = tf.keras.applications.VGG19()
model.summary()

### Model definition 

from tensorflow.keras import regularizers
num_classes = 22
epochs = 100 #examples

# VGG19
# 
VGG19_model = tf.keras.applications.VGG19(input_shape=(28*k,28*k,3),
                                          include_top=False,
                                          weights='imagenet')

print(len(VGG19_model.layers))

#Freezing the first 6 layers to avoid re-training them 

for layer in VGG19_model.layers[:6]:
  layer.trainable = False

# New empty model 
model = tf.keras.Sequential()

# Adding pre-trained model as a layer
model.add(VGG19_model)

# Added layers for fine-tuning 
model.add(Flatten())
model.add(Dropout(0.25))
model.add(Dense(64, kernel_regularizer=regularizers.l2(0.01), activation = 'relu'))
model.add(Dropout(0.25))
model.add(Dense(num_classes, activation = 'softmax'))

### Train the added layers  

model.compile(loss="categorical_crossentropy", 
              optimizer= SGD(learning_rate=0.01),
              metrics=['accuracy'])

from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import ReduceLROnPlateau                                                                              

checkpointer = ModelCheckpoint(filepath='choose/your_predefined/path/for_storage', verbose=1, save_best_only=True,
                               monitor = 'val_acc', mode = 'max')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, 
                              patience=3, min_lr=0.000001)

history= model.fit_generator(train_generator,
                             validation_data = valid_generator, 
                             callbacks=[reduce_lr, checkpointer], 
                             epochs=epochs)

### Store the trained model 
model.save("file_to_folder/store/model.h5")

### Plot the accuracy and loss curves 

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title("Accuracy")
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend(['train','test'])
plt.show()


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title("Loss")
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train','test'])
plt.show()


### Test the model 
Y_pred = model.predict_generator(test_generator, 238 // 32+1)         #indicating the number of elements and the batch size
y_pred = np.argmax(Y_pred, axis=1)

#create labels correspinding to letters of the classes (initially in int)
labels= test_generator.class_indices.keys()    

### Classification report 

from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

print(classification_report(test_generator.classes, y_pred, target_names=['A', 'B', 'C', 'D', 'E', 'F', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'T', 'U', 'V', 'W', 'X', 'Y']))

### Confusion matrix 

cm = confusion_matrix(test_generator.classes, y_pred)

#Plot 
plt.figure()
plt.imshow(cm)
plt.show()


