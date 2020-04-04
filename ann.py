# Artifical neural networks

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('main.csv')
X = dataset.iloc[:, 3 : 13].values
y = dataset.iloc[:, 13].values

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X1 = LabelEncoder()
X[:, 1] = labelencoder_X1.fit_transform(X[:, 1])
labelencoder_X2 = LabelEncoder()
X[:, 2] = labelencoder_X2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[: , 1 :]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

import keras
from keras.models import Sequential
from keras.layers import Dense

# Intialise the neural network
# 2 ways to define a neural network. One is by describing the ANN model
# the other way is by building a graph

# Initialise the ANN
classifier = Sequential() 
# Rectifier function will be our activation function
# Sigmoid function will be chosen for output error
### 6 has been chosen because that is the average between in the nodes in input
### input later and output layer
### Next hidden layer will know what to expect after we add the input dimensions
### in the first hidden layer

# Adding the input and first hidden layer
classifier.add(Dense(6 , kernel_initializer = 'uniform' , activation = 'relu' , input_dim = 11))
# Adding the second layer now
classifier.add(Dense(6 , kernel_initializer = 'uniform' , activation = 'relu'))
# Adding the output layer now
classifier.add(Dense(1 , kernel_initializer = 'uniform' , activation = 'sigmoid')) 
# softmax activation function should have been chosen for multi-dimensional inputs

# Adam is a very efficient stochastic gradient descent algorithm
classifier.compile(optimizer = 'adam' , loss = 'binary_crossentropy' , metrics = ['accuracy'])

classifier.fit(X_train , y_train , batch_size = 10 , epochs = 100)
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
total = 2000
accuracy = (cm[0][0] + cm[1][1]) / total



