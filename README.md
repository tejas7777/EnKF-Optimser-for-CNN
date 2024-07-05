## EnKF Optimizer for CNNs
This repository contains an implementation of an Ensemble Kalman Filter (EnKF) based optimizer for training Convolutional Neural Networks (CNNs). The EnKF optimizer offers a gradient-free approach to training, making it suitable for tasks like image denoising where backpropagation can be computationally expensive.

## Features
EnKF Optimizer: Implements the Ensemble Kalman Filter for gradient-free optimization.
CNN Training: Supports training of Convolutional Neural Networks for tasks such as image denoising.
Parallel Processing: Utilizes PyTorch’s capabilities for efficient parallel processing on both CPU and GPU.
Custom Model Integration: Easily integrate your own CNN models and datasets.
