# -*- coding: utf-8 -*-
"""
Created on Sun Mar  9 16:56:22 2025

@author: rhdut
"""

# main.py
import argparse
from data_preprocessing import load_data
from model import model_create
from train import train_model
from evaluate import eval_model
import time

def args_parse():
    parser=argparse.ArgumentParser(description="Image Classification Script")
    parser.add_argument('--train_dir', type=str, required=True, help='Path to the training data')
    parser.add_argument('--valid_dir', type=str, required=True, help='Path to the validation data')
    parser.add_argument('--test_dir', type=str, required=True, help="Path to the test data")
    parser.add_argument('--img_size', type=int, default=224, help='Size of images form resizing')
    parser.add_argument('--channels', type=int, default=3, help='Number of image channels')
    return parser.parse_args()
    
    
def main():
    # Parse arguments
    args=args_parse()
    print("Arguments parsed:", args)
    
    # Load data
    print(f"Loading data from {args.train_dir}...")
    train_generator, valid_generator, test_generator=load_data(args.train_dir, args.valid_dir, args.test_dir, args.img_size)
    print("Data loaded successfully!")
    
    # Create model
    print("Creating the model...")
    model=model_create(img_size=args.img_size, channels=args.channels, train_generator=train_generator)
    print("Model created successfully")
    
    # Training with progress bar
    print("Starting model training...")
    start_time=time.time()
    
    train_model(model, train_generator, valid_generator, epochs=20)
    end_time=time.time()
    print(f"Training completed in {end_time - start_time:.2f} seconds!")
    
    # Save the trained model
    model_save_path="models\\tflearn_model_v1.h5"
    model.save(model_save_path)
    
    # Evaluate model
    print("Evaluating the model...")
    eval_model(model, test_generator)
    print("Model evaluation completed!")
    
if __name__=="__main__":
    main()