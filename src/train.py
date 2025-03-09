# -*- coding: utf-8 -*-
"""
Created on Sun Mar  9 16:53:10 2025

@author: rhdut
"""

# train.py

def train_model(model, train_generator, valid_generator, epochs=10):
    history=model.fit(train_generator, validation_data=valid_generator, epochs=epochs)
    return history

