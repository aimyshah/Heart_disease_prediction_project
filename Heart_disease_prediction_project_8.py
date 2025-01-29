#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
# Importing various Classification models and then selecting the best one for this dataset using cross validation.
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm


# ### Data collection and processing

# In[2]:


heart_data = pd.read_csv('data(Heart_disease_prediction_project_#8)\heart_disease_data.csv')


# In[3]:


# Printing first five rows.
heart_data.head()


# In[4]:


heart_data.shape


# In[5]:


# Getting basic info about the dataset.
heart_data.info()


# In[6]:


# Getting statistical measures of the dataset
heart_data.describe()


# In[7]:


# Checking the distribution of the 'target' feature to prevent underfitting or overfitting.
heart_data['target'].value_counts()


# In[8]:


# Inference: Here the distribution is almost the same for Having a disease and for not having it.
# 0 --> Healthy Heart
# 1 --> Defective Heart


# #### Splitting the features and Target

# In[9]:


X = heart_data.drop(columns='target')
Y = heart_data['target']


# In[10]:


print(X)
print(Y)


# In[11]:


# List of models.
models = [LogisticRegression(max_iter=1000), svm.SVC(kernel='linear'), KNeighborsClassifier()]


# In[12]:


# Making a function for calculating accuracy of each model using cross validation.
def compare_models_cross_validation():
    
    for model in models:
        
        cv_score = cross_val_score(model, X, Y, cv=5)
        
        mean_accuracy = sum(cv_score)/len(cv_score)
        
        mean_accuracy = mean_accuracy*100
        
        mean_accuracy = round(mean_accuracy, 2)
        
        print('Cross Validation accuracies for', model, ': ', cv_score)
        
        print('Accuracy in percentage of', model, 'is: ', mean_accuracy)
        
        print('-----------------------------------------------------------')


# In[13]:


compare_models_cross_validation()


# In[14]:


# Inference: Accuracy of SVM and LogisticRegression is the same. Using LogisticRegression. KNeighborsClassifier is not performing well enough


# #### Splitting the data into Training and Test data

# In[15]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y)


# ### Model Training

# #### LogisticRegression

# In[16]:


model = LogisticRegression(max_iter=1000)


# In[17]:


model.fit(X_train, Y_train)


# ### Model Evaluation

# In[18]:


# Accuracy on training data.
X_train_predictions = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_predictions, Y_train)
print('Accuracy on Training data: ', round(training_data_accuracy * 100, 2))


# In[19]:


# Accuracy on testing data.
X_test_predictions = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_predictions, Y_test)
print('Accuracy on test data: ', round(test_data_accuracy * 100, 2))


# In[20]:


# Inference: So the model is not Overfitting to the training data as the accuracy scores are almost the same.


# ### Building a Predictive system

# In[24]:


# Define the feature names.
feature_names = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"]

# Function for getting user input.
def get_user_input():
    print("Enter the values of the following features: ")

    input_data = []
    for feature in feature_names:
        value = float(input(f"{feature}: "))
        input_data.append(value)

    # Convert the input to a numpy array and reshape it to (1, -1)
    return np.array(input_data).reshape(1, -1)
    
# Get user input
input_data = get_user_input()

# Make a prediction using the model.
prediction = model.predict(input_data)

# Display the result.
if prediction[0] == 0:
    print("The Person does not have a heart disease")
else:
    print("The Person has a heart disease")

