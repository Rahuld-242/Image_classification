# -*- coding: utf-8 -*-
"""
Created on Sun Mar  9 16:32:06 2025

@author: rhdut
"""

from tensorflow.keras import layers,Sequential,models
from keras.applications import MobileNetV2

def model_create(img_size, channels, train_generator):
    """
    Create and return a MobileNetV2-based model with the final layer dynamically
    set based on the number of classes in the train_generator
    
    Parameters:
    - img_size: Integer, the target image size (height and width)
    - channels: Integer, the number of channels in the image (3 for RGB)
    - train_generator: The generator used to load the training data (for getting the number of classes)
    
    Returns:
    -model: A compiled Keras Sequential model
    """
    # Define the base model (MobileNetV2) without the top classification layers
    base_model=MobileNetV2(weights='imagenet',include_top=False, input_shape=(img_size,img_size,channels))
    base_model.trainable=False #Freeze the layers of MobileNet
    
    # Create the model
    model=Sequential()
    
    # Add the base model
    model.add(base_model)
    
    # Add a flatten layer to convert 2D feature maps to 1D vector
    model.add(layers.Flatten())
    
    # Add a fully connected layer with 512 neurons
    model.add(layers.Dense(512,activation='relu'))
    
    # Dynamically set the number of classes for the output layer
    num_classes=len(train_generator.class_indices) # Number of classes from the training data
    model.add(layers.Dense(num_classes,activation='softmax')) # Output layer
    
    # Compile the model
    model.compile(optimizer='adam',loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model
    
    