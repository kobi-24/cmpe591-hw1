# CMPE 591 - Homework 1: Deep Learning for Robotics

This repository contains the complete structure which is needed in Homework 1, including data generation, Multi-Layer Perceptron (MLP), Convolutional Neural Network (CNN), and also an Image-to-Image Predictor.

## Prerequisites
As prerequisite, physics environment and PyTorch installation is required . The code automatically detects and utilizes CUDA if available for hardvare acceleration.

## Phase 1: Data Generation
To generate the 1000 required samples as Initial State, Action, Final Position and Final State Image, run the following command:
```bash
python generate_data.py

![Next Frame Prediction Results](deliverable3_results.png)