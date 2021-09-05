# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 11:42:16 2020

@author: Behsa PC1
"""

import os
import numpy
from keras.applications.mobilenet import MobileNet
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
#******************reading data***************************

train_crazing_dir = os.path.join('F:/mostafaie/cold-strip/database/NEU-DET/train/images/crazing')
train_inclusion_dir = os.path.join('F:/mostafaie/cold-strip/database/NEU-DET/train/images/inclusion/')
train_patches_dir = os.path.join('F:/mostafaie/cold-strip/database/NEU-DET/train/images/patches/')
train_pitted_surface_dir = os.path.join('F:/mostafaie/cold-strip/database/NEU-DET/train/images/pitted_surface/')
train_rolledin_scale_dir = os.path.join('F:/mostafaie/cold-strip/database/NEU-DET/train/images/rolled-in_scale/')
train_scratches_dir = os.path.join('F:/mostafaie/cold-strip/database/NEU-DET/train/images/scratches/')

train_crazing_names = os.listdir(train_crazing_dir)
train_inclusion_names = os.listdir(train_inclusion_dir)
train_patches_names = os.listdir(train_patches_dir)
train_pitted_surface_names = os.listdir(train_pitted_surface_dir)
train_rolledin_scale_names = os.listdir(train_rolledin_scale_dir)
train_scratches_names = os.listdir(train_scratches_dir)

print('total training crazing images:', len(train_crazing_names))
print('total training inclusion images:', len(train_inclusion_names))
print('total training patches images:', len(train_patches_names))
print('total training pitted_surface images:', len(train_pitted_surface_names))
print('total training rolled_in_scale images:', len(train_rolledin_scale_names))
print('total training scratches images:', len(train_scratches_names))


# Rescaling images
train_datagen = ImageDataGenerator(rescale=1/255,validation_split=0.1)
test_datagen = ImageDataGenerator(rescale=1/255)

train_generator = train_datagen.flow_from_directory(
    directory = 'F:/mostafaie/cold-strip/database/NEU-DET/train/images/',
    target_size = (224, 224),
    batch_size = 32,
    class_mode = 'categorical',
    subset='training'
)

validation_generator  = train_datagen.flow_from_directory(
    directory = 'F:/mostafaie/cold-strip/database/NEU-DET/train/images/',
    target_size = (224, 224),
    batch_size = 32,
    class_mode = 'categorical',
    subset='validation'
)


test_generator = test_datagen.flow_from_directory(
    directory = 'F:/mostafaie/cold-strip/database/NEU-DET/validation/images/',
    target_size = (224, 224),
    batch_size = 32,
    class_mode = 'categorical', shuffle=False)

#**********************create model*******************************

# parameters for architecture
input_shape = (224, 224, 3)
num_classes = 6
conv_size = 32

# parameters for training
num_epochs = 9

# load MobileNet from Keras
MobileNet_model = MobileNet(include_top=False, input_shape=input_shape)

# add custom Layers
x = MobileNet_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation="relu")(x)
Custom_Output = Dense(num_classes, activation='softmax')(x)

# define the input and output of the model
model = Model(inputs = MobileNet_model.input, outputs = Custom_Output)

# compile the model
model.compile(loss='categorical_crossentropy',
                    optimizer='adam',
                    metrics=['accuracy'])

model.summary()


#***************train**********************************************

# # train the model 
# history= model.fit(
#     train_generator,
#     epochs=num_epochs,
#     verbose=1,
#     validation_data=validation_generator
# )



# model.save_weights("model.h5")
# #************plot hist******************
# # summarize history for accuracy
# plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()
# # summarize history for loss
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()




#************************************test**********************************
model.load_weights("model.h5")
test_loss,test_acc=model.evaluate (test_generator)
print("test acc is: ",test_acc)

#prdict one img
x_test=mpimg.imread('F:\\mostafaie\\cold-strip\\database\\NEU-DET\\validation\\images\\inclusion\\inclusion_297.jpg')
x_test=x_test/255
x_test=numpy.resize(x_test,(224,224,3))
x_test = numpy.expand_dims(x_test, axis=0)
prediction=model.predict(x_test)