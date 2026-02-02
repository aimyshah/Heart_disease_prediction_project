## Table of Contents
- [Project Overview](#project-overview)  
- [Problem Statement](#problem-statement)  
- [Dataset](#dataset)  
  - [Dataset Overview](#dataset-overview)  
  - [Dataset Source](#dataset-source)  
  - [Dataset Size](#dataset-size)  
  - [Feature Description](#feature-description)  
  - [Target Variable](#target-variable)  
- [Approach](#approach)  
- [Results](#results)  

---

## Project Overview

This project uses machine learning to predict whether a person has heart disease or not. It works with several medical and health-related parameters that influence the likelihood of heart disease, such as age, resting blood pressure, cholesterol level, fasting blood sugar, and more.  

The main goal of this project is early detection of heart disease using patient medical data and history. Early prediction can help in taking preventive measures before the condition becomes critical.  

---

## Problem Statement

Heart disease is one of the leading causes of death worldwide. A major challenge is that heart-related problems often do not show clear symptoms in the early stages, making them difficult to detect without proper medical examination.  

Many people remain unaware of their condition until it becomes severe, sometimes resulting in sudden heart attacks or other life-threatening situations. Early detection of heart disease can save lives, and the application of machine learning in the medical field can play a crucial role in improving healthcare outcomes.  

---

## Dataset

### Dataset Overview

This project uses the **Heart Disease Dataset**, a well-known dataset commonly used for classification tasks in healthcare-related machine learning projects.  

### Dataset Source

The dataset was downloaded from **Kaggle**, where it is shared as the *Heart Disease Dataset*.  

### Dataset Size

- **Number of instances:** 304  
- **Number of features:** 13 independent variables  
- **Target variable:** 1  

### Feature Description

The dataset includes the following input features:  

- **age:** Age of the patient  
- **sex:** Gender of the patient  
- **cp:** Chest pain type (4 values)  
- **trestbps:** Resting blood pressure  
- **chol:** Serum cholesterol in mg/dl  
- **fbs:** Fasting blood sugar > 120 mg/dl  
- **restecg:** Resting electrocardiographic results (values 0, 1, 2)  
- **thalach:** Maximum heart rate achieved  
- **exang:** Exercise-induced angina  
- **oldpeak:** ST depression induced by exercise relative to rest  
- **slope:** Slope of the peak exercise ST segment  
- **ca:** Number of major vessels (0–3) colored by fluoroscopy  
- **thal:** 0 = normal, 1 = fixed defect, 2 = reversible defect  

### Target Variable

- **target:**  
  - `1` = Has heart disease  
  - `0` = Does not have heart disease  

---

## Approach

1. Loaded and explored the dataset.  
2. Preprocessed the data and split it into training and testing sets.  
3. Used cross-validation to select the best-performing classification model.  
4. Trained a **Logistic Regression** model.  
5. Evaluated the model using the **accuracy score**.  
6. Built a prediction function that takes user input features and predicts whether the person has heart disease or not.  

---

## Results

- **Training Accuracy:** 86.57%  
- **Testing Accuracy:** 85.70%  
