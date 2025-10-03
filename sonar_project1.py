import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#loading the dataset to a pandas dataframe
sonar_data = pd.read_csv('/sonar data.csv', header=None)

sonar_data.head()

#number of rows and columns
sonar_data.shape

sonar_data.describe() #describe ---> statistical measures of the data

sonar_data[60].value_counts()

# M-->Mine
# R-->Rock

sonar_data.groupby(60).mean()

#separating data and labels
X = sonar_data.drop(columns=60, axis=1)
Y = sonar_data[60]

print(X)
print(Y)

#Training and Test data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.1, stratify=Y, random_state=1)

print(X.shape, X_train.shape, X_test.shape)

print(X_train)
print(Y_train)

#Model Training-->Logistic Regression
model = LogisticRegression()

#training the logistic Regression model with training data
model.fit(X_train, Y_train)

#Model Evaluation
#accuracy on the training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

print('Accuracy on training data', training_data_accuracy)

#accuracy on the test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

print('Accuracy on test data', test_data_accuracy)

#Making a predictive System
input_data = (0.0286,0.0453,0.0277,0.0174,0.0384,0.0990,0.1201,0.1833,0.2105,0.3039,0.2988,0.4250,0.6343,0.8198,1.0000,0.9988,0.95) # A truncated tuple was provided in the source

# changing the input_data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

#reshape the np array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = model.predict(input_data_reshaped)
print(prediction)

if(prediction[0]=='R'):
    print('The object is a Rock')
else:
    print('The object is a mine')
