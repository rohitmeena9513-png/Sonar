

# Sonar Rock vs. Mine Prediction  sonar-prediction

This project uses a **Logistic Regression** model to classify sonar signals and predict whether an object is a Rock ('R') or a Mine ('M'). It serves as a basic example of a complete machine learning workflow, from data loading and analysis to model training, evaluation, and prediction.

## Overview 🔎

The goal is to build a binary classification model that can distinguish between metal cylinders (mines) and cylindrical rocks based on sonar returns. The Python script covers all the essential steps:

  - Loading the dataset using Pandas.
  - Performing basic exploratory data analysis.
  - Separating features and labels.
  - Splitting the data into training and testing sets.
  - Training a Logistic Regression model.
  - Evaluating the model's accuracy.
  - Creating a predictive system to classify new data.

-----

## Dataset 📊

The project utilizes the **"Sonar, Mines vs. Rocks"** dataset.

  - **Source**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Connectionist+Bench+\(Sonar,+Mines+vs.+Rocks\))
  - **Instances**: 208
  - **Features**: 60 numeric features, representing the energy of the sonar signal at different angles.
  - **Target Variable**: A single label indicating the object type ('R' for Rock, 'M' for Mine).

The dataset is loaded from a CSV file named `sonar data.csv`.

-----

## Workflow ⚙️

The code follows a standard machine learning pipeline:

1.  **Data Loading and Inspection**: The dataset is loaded into a Pandas DataFrame. Initial analysis is done using `.head()`, `.shape`, and `.describe()` to understand its structure and statistical properties.
2.  **Data Preprocessing**: The features (X) and the target labels (Y) are separated.
3.  **Train-Test Split**: The data is split into a training set (90%) and a testing set (10%). `stratify=Y` is used to ensure that both the training and test sets have a proportional representation of 'Rock' and 'Mine' classes.
4.  **Model Training**: A `LogisticRegression` model from Scikit-learn is instantiated and trained on the training data using the `.fit()` method.
5.  **Model Evaluation**: The model's performance is evaluated by calculating the accuracy score on both the training data and the test data to check for overfitting.
6.  **Predictive System**: A simple function is created to take a new set of 60 sonar readings, reshape it, and use the trained model to predict whether the object is a Rock or a Mine.

-----

## How to Use 🚀

To run this project on your local machine, follow these steps:

### 1\. Prerequisites

Make sure you have Python installed. You will need the following libraries:

  - NumPy
  - Pandas
  - Scikit-learn

### 2\. Installation

Clone this repository to your local machine:

```bash
git clone <your-repository-url>
cd <repository-directory>
```

Install the required libraries using pip:

```bash
pip install numpy pandas scikit-learn
```

### 3\. Get the Dataset

Download the dataset from the [UCI repository](https://www.google.com/search?q=https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/sonar/sonar.all-data) and save it as `sonar data.csv` in the root directory of the project.

### 4\. Run the Script

Execute the Python script from your terminal:

```bash
python sonar_classifier.py
```

*(Assuming you save the code as `sonar_classifier.py`)*

The script will print the model's accuracy on the training and test data, followed by a prediction for a sample input.

-----

## Example Prediction 🎯

The script includes a function to predict the class for a new data point. The input data should be a tuple or list containing 60 numeric values.

```python
#Making a predictive System
# Note: The input tuple must contain 60 feature values.
input_data = (0.0270,0.0092,0.0145,0.0278,0.0412,0.0757,0.1026,0.1138,0.0794,0.1520,0.1601,0.2255,0.2843,0.2818,0.3385,0.3739,0.4502,0.5283,0.6122,0.6790,0.7238,0.7818,0.8373,0.8973,0.9423,0.9939,1.0000,0.9333,0.8021,0.6582,0.5283,0.4033,0.3018,0.2333,0.2223,0.2318,0.2603,0.2594,0.2305,0.1775,0.1463,0.1264,0.0818,0.0520,0.0243,0.0189,0.0103,0.0122,0.0104,0.0054,0.0024,0.0019,0.0036,0.0072,0.0054,0.0029,0.0023,0.0019,0.0011,0.0012)

# changing the input_data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

#reshape the np array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = model.predict(input_data_reshaped)
print(prediction)

if(prediction[0]=='R'):
    print('The object is a Rock')
else:
    print('The object is a Mine')
```

This will output the prediction based on the trained model.
