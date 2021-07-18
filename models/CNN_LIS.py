### Import necessary modules 

import keras
import os
import numpy
import time
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from keras.optimizers import Adam

from keras.callbacks import EarlyStopping

### Model instantiation 

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Activation, Dropout, Flatten, Dense

# model istantiation
def istantiate_model(input_shape, print_bool = True):
    model = Sequential()
    #1st
    model.add(Conv2D(64, kernel_size=(3, 3),padding='same',input_shape=input_shape))
    model.add(Conv2D(64, kernel_size=(3, 3),padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    #2nd
    model.add(Conv2D(64, kernel_size=(3, 3),padding='same',input_shape=input_shape))
    model.add(Conv2D(64, kernel_size=(3, 3),padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    #3d
    model.add(Conv2D(64, kernel_size=(3, 3),padding='same',input_shape=input_shape))
    model.add(Conv2D(64, kernel_size=(3, 3),padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    #4th
    model.add(Flatten())
    model.add(Dense(128))
    model.add(Dropout(0.5))

    #5th
    model.add(Dense(128))
    model.add(Dropout(0.5))

    #6th
    model.add(Dense(22))
    model.add(Activation('softmax'))

    if (print_bool != False):
        model.summary()

    return model
  
  
 ### Define labels and image-size 

# classes' labels
sign_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'T', 'U', 'V', 'W', 'X', 'Y']  
#non-static signs excluded 

# size of our images.
img_width, img_height = 64, 64

### Dataset folder (images must be stored in sub-folders per letter)

DATASET_FOLDER= "path/to_images/general_folder" 


#partition folders 
train_data_dir = DATASET_FOLDER + "/train"
validation_data_dir = DATASET_FOLDER + "/validation"
testing_data_dir = DATASET_FOLDER + "/testing"


###  Epochs and batch-size definition  
epochs = 100                   
batch_size = 32            

test_type = "_noBatchNormalization_"

# detect img data format
if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)
    

model = istantiate_model(input_shape)


### Model compiling 

model.compile(loss='categorical_crossentropy',
              optimizer='adam'  ,
              metrics=['accuracy'])

### Data Augmentation 

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=40,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=False)

test_datagen = ImageDataGenerator(rescale=1. / 255,
    rotation_range=40,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=False)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

testing_generator = test_datagen.flow_from_directory(
    testing_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

### Train the model 
early_stopping = EarlyStopping(monitor='val_loss', patience=2)

history = model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    verbose=1,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size)

model.save("file_to_folder/store/model.h5")


### Plot the curves for accuracy and loss 

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title("Accuracy")
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend(['train','validation'])
plt.show()


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title("Loss")
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train','validation'])
plt.show()

### Extract evaluation scores  

score = model.evaluate_generator(validation_generator, nb_validation_samples//batch_size, workers=12, use_multiprocessing=False)
print("Evaluate generator results:")
print("Evaluate loss: " + str(score[0]))
print ("Evaluate accuracy: " + str(score[1]))


### Evaluation on a pre-defined test set 

sign_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'T', 'U', 'V', 'W', 'X', 'Y']
index_labels = [0, 1, 2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]

### Model predictions 
Y_pred = model.predict_generator(testing_generator, NB_SAMPLES // 64 + 1  )
y_pred = np.argmax(Y_pred, axis=1)

### Generate confusion matrix 
cm = confusion_matrix(testing_generator.classes, y_pred, index_labels)

### Print classification report 
print(classification_report(testing_generator.classes, y_pred, target_names=sign_labels))
  
  
  
