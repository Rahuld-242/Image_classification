# Sports Image Classification

This project demonstrates how to perform **image classification** using **transfer learning** with the **MobileNetV2** model. The script is designed to classify images in multiple categories, such as various sports, using a **pretrained MobileNetV2 model**. It incorporates data preprocessing, model training, and evaluation, along with version control practices and file management.

## Table of Contents
- [Overview](#overview)
- [Setup](#setup)

## Overview
This repository provides a Python-based solution for classifying images of various sports categories using **TensorFlow** and **Keras**. We utilize a **MobileNetV2** pretrained model for transfer learning, which significantly reduces the time required to train the model by leveraging pre-learned features from the ImageNet dataset.

### Key Features:
- **Transfer Learning** with MobileNetV2.
- **Data Augmentation** for better model generalization.
- **Image Preprocessing** including resizing and normalization.
- **Customizable Model Parameters**: Can handle different image sizes, channels, and the number of categories.
- **Version Control**: Git is used for version control, and large files are managed using `.gitignore` and BFG Repo-Cleaner to avoid pushing unnecessary files.

---

## Setup

### Prerequisites
Before running this script, ensure that you have the following dependencies installed:

- Python 3.x
- TensorFlow 2.x
- Keras
- NumPy
- Matplotlib

You can install the necessary dependencies by using the following command:

```bash
pip install -r requirements.txt
