# Detecting-Spam-Emails-Using-Tensorflow-in-Python



Email Spam Classification using LSTM
Overview


This repository contains code for building a model to classify emails as spam or non-spam (ham). The model utilizes a Long Short-Term Memory (LSTM) neural network for sequence processing. The dataset used for training and testing is a collection of emails stored in a CSV file.

Prerequisites
Make sure you have the following libraries installed:

NumPy
Pandas
Matplotlib
Seaborn
NLTK
String
TensorFlow
WordCloud
Scikit-Learn
You can install them using the following:

bash
Copy code
pip install numpy pandas matplotlib seaborn nltk tensorflow wordcloud scikit-learn
Dataset
The dataset is stored in a CSV file, and the file path needs to be specified in the code. You can replace the file_path variable with the path to your CSV file.

Exploratory Data Analysis (EDA)
The code includes exploratory data analysis to visualize the distribution of spam and non-spam emails in the dataset.
Downsampling is performed to balance the dataset.
Text Preprocessing
Text preprocessing is carried out to clean and prepare the text data for model training.
Punctuation is removed, and stopwords are eliminated from the text.
Word Clouds
Word clouds are generated to visually represent the most frequent words in spam and non-spam emails.
Model Building and Training
The text data is tokenized and converted into sequences for model input.
An LSTM-based neural network is constructed for email classification.
The model is trained using a binary cross-entropy loss function and the Adam optimizer.
Early stopping and learning rate reduction on plateau are employed as callbacks during training.
Model Evaluation
The trained model is evaluated on the test set.
Test loss and accuracy are printed.
A plot of training and validation accuracy over epochs is displayed.
