# -*- coding: utf-8 -*-
"""
Created on Sun Mar  9 16:54:49 2025

@author: rhdut
"""

# evalutate.py

def eval_model(model, test_generator):
    loss, accuracy=model.evaluate(test_generator)
    print(f"Test Loss: {loss}")
    print(f"Test Accuracy: {accuracy}")