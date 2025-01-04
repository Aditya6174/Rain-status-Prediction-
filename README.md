# Weather Prediction Using Machine Learning

## Overview
This repository contains a Python-based machine learning project aimed at predicting whether it will rain tomorrow (`RainTomorrow`) based on weather data from Australia. It leverages data preprocessing, exploratory data analysis, and logistic regression to achieve this goal.

## Features
- Data cleaning and preprocessing, including handling missing values and scaling features.
- Exploratory data analysis using Seaborn, Matplotlib, and Plotly.
- Training and evaluation of a logistic regression model.
- Feature encoding using one-hot encoding for categorical variables.
- Accuracy evaluation with confusion matrices and visualizations.

## Dataset
The dataset `weatherAUS.csv` is sourced from Kaggle and contains the following key attributes:
- Weather parameters such as temperature, humidity, pressure, etc.
- Target variable: `RainTomorrow`.

### Data Preprocessing Steps
1. **Handling Missing Values**: Imputation using mean for numeric features.
2. **Feature Scaling**: Min-max normalization for numeric columns.
3. **Categorical Encoding**: One-hot encoding for non-numeric data.
4. **Data Splitting**: Train (2010-2014), Validation (2015), and Test sets (2016 onwards).

## Requirements
To run the code, install the required Python packages using:

```bash
pip install -r requirements.txt
