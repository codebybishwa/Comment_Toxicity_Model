# Toxic Comment Classification

## Overview
This project implements a deep learning model to classify comments as toxic, hateful, threatening, or neutral. The model is built using TensorFlow and Keras, leveraging a dataset originally sourced from Kaggle. The goal is to identify harmful comments and promote a safer online environment.

## Model Architecture
The model is built using a Sequential architecture with the following layers:
1. Embedding Layer: Encodes the input text into dense vectors.
2. Bidirectional LSTM Layer: Processes sequences in both forward and backward directions to capture context.
3. Dense Layers: Fully connected layers for feature extraction.
4. Output Layer: Produces probabilities for each class using a sigmoid activation function.

## Process
- Loading the Dataset: The dataset is loaded into a Pandas DataFrame for preprocessing.
- Preprocessing: Text vectorization is performed to convert comments into numerical format.
- Tokenization is applied to prepare the text for the model.
- The dataset is split into training and testing sets.
- Model Creation: A Sequential model is created and compiled with binary cross-entropy loss and the Adam optimizer.
- Training: The model is trained on the training dataset.
- Evaluation: The model's performance is evaluated on the test dataset.

## Gradio App
At the end of the project, a Gradio app is developed, allowing users to input comments and receive predictions on whether the comment is toxic, hateful, or threatening.

## Tech Stacks Used
- Python: Programming language used for the implementation.
- TensorFlow: Framework for building and training the deep learning model.
- Keras: High-level API for building neural networks.
- Pandas: Library for data manipulation and analysis.
- Gradio: Library for creating user interfaces for machine learning models.
