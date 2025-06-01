# HeartDisease: Exploratory Data Analysis and Visualization of Heart Disease Risk Prediction

Heart disease remains a leading cause of death globally, necessitating effective prediction tools to mitigate its impact. This study focuses on developing an optimized heart disease risk prediction model using the Heart Disease UCI dataset. Key methodologies include employing XGBoost for classification, enhanced with hyperparameter tuning via Optuna to maximize performance. Data preprocessing steps involved handling missing values, feature standardization, and addressing class imbalance using SMOTE. To ensure explainability, SHAP plots were utilized to interpret the model’s predictions. The model achieved a high ROC-AUC score of 0.934 and F1-score of 0.851, outperforming baseline methods. Finally, a user-friendly application was deployed using Streamlit to enable practical usage of the prediction tool, demonstrating potential to alleviate healthcare burdens through interpretable and efficient prediction techniques.

## Dataset
The dataset used in this study is the Heart Disease UCI dataset from the UCI Repository (https://archive.ics.uci.edu/dataset/45/heart+disease), comprising 303 instances with 14 attributes that describe patient characteristics such as age, sex, and cholesterol levels.

<img width="383" alt="image" src="https://github.com/user-attachments/assets/695f4386-65ad-43c2-bed4-bd8e1ce2a9e2" />

To gain an initial understanding of the dataset, we computed descriptive statistics for all features. A summary of these descriptive statistics is presented as follow:

<img width="388" alt="image" src="https://github.com/user-attachments/assets/34d240bd-818d-4f1c-89ce-514ac93ba3cc" />

### Visualization
To have a rough idea of how the data is distributed, we made some bar plots and a correlation heatmap.

<img width="406" alt="image" src="https://github.com/user-attachments/assets/03d50f2c-3728-4237-b00e-4235635c1fc9" />

<img width="374" alt="image" src="https://github.com/user-attachments/assets/69e14f49-ab3f-41fc-ba6e-c7ef63a02f7b" />

<img width="297" alt="image" src="https://github.com/user-attachments/assets/52f155c5-e073-4000-9cb3-4374100798ca" />

## Feature Engineering

* Handling Missing Values:
First, we address missing values, which can skew our analysis and model performance. For numerical variables, we used the median to fill in the missing values. This is a robust measure that is less affected by outliers compared to the mean. For categorical variables, we have used the mode, which is the most frequent value in the category.

* Feature Standardization:
Next, we standardized our features to ensure that they are on the same scale. This is important because many machine learning algorithms are sensitive to the scale of the input features. We have used the StandardScaler, which standardizes features by removing the mean and scaling to unit variance.

* New Features Creation:
Finally, we created 4 new features, as shown in table 3, to capture additional information and relationships that may improve the modeling performance.

<img width="394" alt="image" src="https://github.com/user-attachments/assets/1e16cc19-fd1f-4d66-9fbb-ac87e5e22dcb" />

## Comparison Analysis of classification methods
<img width="385" alt="image" src="https://github.com/user-attachments/assets/ffb8af68-0b36-4333-9292-f41cd809a31b" />

## Impact of fine-tuning various hyperparameters on model performancce
<img width="321" alt="image" src="https://github.com/user-attachments/assets/28d3a838-9ca8-4045-8ded-0c58449cc40b" />

## Ablation Study
<img width="341" alt="image" src="https://github.com/user-attachments/assets/1607c729-32dc-411e-9300-2310a3931ffe" />

## Streamlit Deployment
<img width="180" alt="image" src="https://github.com/user-attachments/assets/196859f9-fa97-45f8-8bab-714884aee16d" />

<img width="327" alt="image" src="https://github.com/user-attachments/assets/a9441534-d895-4c36-9fd7-e356c8e10a7f" />










