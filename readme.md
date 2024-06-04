# Wine Classification Project

This project aims to classify wines as either red or white using various machine learning algorithms. The dataset used is the Wine dataset.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Preprocessing](#preprocessing)
- [Models and Results](#models-and-results)
- [Conclusion](#conclusion)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Introduction

The goal of this project is to predict whether a given wine is red or white. We applied several machine learning algorithms and evaluated their performance.

## Dataset

The dataset used in this project contains various features of wines, such as acidity, sugar levels, and pH. It is divided into two classes: red wine and white wine.

## Preprocessing

Different preprocessing techniques were applied to the data, including imputation and scaling. The specific preprocessing steps varied for different models to optimize their performance.

## Models and Results

Here are the models used and their respective results:

1. **K-Nearest Neighbors (KNN)**
   - Accuracy: 99.67%
   - Hyperparameters: imputer=1, n_neighbors=3, p=1, weights=distance
   - Best Parameters: `{'p': 1, 'weights': 'distance'}` 
   - Model: `KNeighborsClassifier(n_neighbors=11, p=1, weights='distance')`

2. **Logistic Regression**
   - Accuracy: 97.06%
   - Hyperparameters: `{'C': 10.0, 'penalty': 'l1'}`
   - Model: `LogisticRegression(C=10.0, max_iter=1000000, penalty='l1', solver='saga')`

3. **Naive Bayes Gaussian**
   - Accuracy: 
     - Without Scaling: 96.88%
     - Min-Max Scaling: 97.48%
     - Min-Max Scaling + Power Transformer: 99.02%

4. **Naive Bayes Multinomial**
   - Accuracy:
     - Without Scaling: 92.21%
     - With Scaling: 75.76%

5. **Naive Bayes Bernoulli**
   - Accuracy:
     - Without Scaling: 77.58%
     - With Scaling: 77.62%

6. **Support Vector Machine (SVM)**
   - Accuracy: 99.67%
   - Hyperparameters: `{'C': 1.0, 'gamma': 1.0, 'kernel': 'rbf'}`
   - Model: `SVC(gamma=1.0)`

7. **Decision Tree**
   - Accuracy: 99.67%

## Conclusion

The KNN, SVM, and Decision Tree models achieved the highest accuracy of 99.67% in predicting whether a wine is red or white. The preprocessing techniques such as scaling and power transformation significantly improved the performance of Naive Bayes models.

