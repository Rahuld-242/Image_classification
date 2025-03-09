# -*- coding: utf-8 -*-
"""
Created on Sun Mar  9 12:29:28 2025

@author: rhdut
"""
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_data(train_dir, valid_dir, test_dir, img_size, batch_size=32):
    # Data loading and preprocessing
    train_datagen=ImageDataGenerator(rescale=1./255, rotation_range=30, width_shift_range=0.2,
                                     height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
    valid_datagen=ImageDataGenerator(rescale=1./255)
    test_datagen=ImageDataGenerator(rescale=1./255)
    
    train_generator=train_datagen.flow_from_directory(train_dir, target_size=(img_size,img_size),
                                                      batch_size=batch_size,class_mode='categorical')
    valid_generator=valid_datagen.flow_from_directory(valid_dir, target_size=(img_size,img_size),
                                                      batch_size=batch_size,class_mode='categorical')
    test_generator=test_datagen.flow_from_directory(test_dir, target_size=(img_size,img_size),
                                                    batch_size=batch_size,class_mode='categorical')
    return train_generator, valid_generator, test_generator