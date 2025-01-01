# Immverse Ai Assginment (Voice Classifier)
=====================================
- Text Classification for Active vs. Passive Voice Detection
 
 # Go through the notebook -----(https://github.com/Himanshuu9/Immverseai_Himanshu/blob/main/voice_classification_final.ipynb)

# Active vs Passive Voice Classification

This project involves building a machine learning model to classify sentences into **Active** or **Passive** voice using natural language processing (NLP) techniques. The model leverages various classification algorithms to detect sentence structure and determine the voice.

## Table of Contents

- [Project Overview](#project-overview)
- [Technologies Used](#technologies-used)
- [Dataset](#dataset)
- [Model Training and Evaluation](#model-training-and-evaluation)
- [Results](#results)
- [How to Use](#how-to-use)
- [License](#license)

## Project Overview

This project aims to develop a text classification model that can accurately distinguish between active and passive voice sentences. The model processes input text, extracts relevant features, and classifies the text based on the voice detected.

Key Objectives:
- To preprocess and clean text data.
- To train and evaluate different machine learning models.
- To assess the model's performance using appropriate metrics.
- To provide an explanation of the model's decision-making process (explainability).

## Technologies Used

- **Python**: Programming language used for data processing and model training.
- **scikit-learn**: Library for machine learning algorithms and tools.
- **Pandas**: For data manipulation and analysis.
- **NumPy**: For numerical operations.
- **Matplotlib/Seaborn**: For data visualization.
- **Jupyter Notebook**: For interactive development and testing.

## dataset ------ ( https://github.com/Himanshuu9/Immverseai_Himanshu/blob/main/immverse_ai_eval_dataset.xlsx )

## model-training-and-evaluation 
Basic Overview

Display basic dataset information (column types, null values) and show the first few rows to understand the structure and contents.
- Finding out null values 
- Finding out missing values
- Finding out unique values
- Finding out data types
- Finding out the summary of the data

# Spliting Dataset

Spliting the dataset according to task:
- 60% for training
- 20% for validation
- 20% for testing

Starify:
- Maintains the distribution of the target variable (voice) across splits.

Example:

* Proportional Distribution:

    In the original data:
   - 50% of rows belong to class 'Active'.
   - 50% of rows belong to class 'Passive'.
   - Both train_data and test_data maintain this 50%-50% distribution.
* Without stratify: If you don’t use the stratify parameter, the class distribution in the splits might not match the original dataset, leading to skewed datasets.

## Use of Classification Models

Used Classifier Models for training models:
- Random Forest
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- Decision Tree

# Training & Testing of Models

Training:
- The model is trained on the training dataset (X_train and y_train).

Validation Predictions:
- The model predicts labels for the validation dataset (X_val).
- Calculate performance summary (accuracy, precision, recall, F1-score) for validation predictions.

Testing:
- The model predicts labels for the test dataset (X_test).
- Calculate performance summary (accuracy, precision, recall, F1-score) for test predictions.

Confusion Matrix Visualization:
- Plot a confusion matrix for the test predictions.

# results

Model Comparison Results:
                    Model  Accuracy  Precision  Recall  F1 Score
0           Random Forest     1.000        1.0   1.000  1.000000
1     Logistic Regression     0.500        0.5   0.500  0.466667
2     K-Nearest Neighbors     1.000        1.0   1.000  1.000000
3  Support Vector Machine     0.125        0.1   0.125  0.111111
4           Decision Tree     1.000        1.0   1.000  1.000000


![result_plot](https://github.com/user-attachments/assets/95bab935-443f-4c88-8b5d-d6210b71d102)



## Analyse the Model based on metrics

The model is trained on the dataset and the metrics are calculated. The metrics are used to evaluate the performance of the model.

So, according to the metrics I found that 3 models gives the same performance which are:
- Random Forest
- Support Vector Machine (SVM)
- Decision Tree

But I choose best model for this classification is: Random Forest

- Better generalization than a single Decision Tree due to its ensemble nature, reducing overfitting.
- Higher accuracy and robustness, handling noisy data well.
- Feature importance insights for better interpretability.
- Scalability and efficiency with large datasets, and it handles both categorical and continuous data well.
- Requires less tuning and works effectively without needing feature scaling.
- It's a solid, versatile choice when compared to Decision Trees and SVM.

# Prediction of voice over sentences

The following is an example of how to use the model to predict the voice over sentences:

Enter any of the sentences in the input:

(Examples)

1. * Active voice: "I am cooking a meal"
   * Passive voice: "A meal is being cooked by me"

2. * Active voice: "He writes an essay"
   * Passive voice: "An essay is written by him"

3. * Active voice: "Carmen sings the song"
   * Passive voice: "The song is sung by Carmen"

  
