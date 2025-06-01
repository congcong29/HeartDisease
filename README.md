# HeartDisease
Exploratory Data Analysis and Visualization of Heart Disease Risk Prediction

Heart disease remains a leading cause of death globally, necessitating effective prediction tools to mitigate its impact. This study focuses on developing an optimized heart disease risk prediction model using the Heart Disease UCI dataset. Key methodologies include employing XGBoost for classification, enhanced with hyperparameter tuning via Optuna to maximize performance. Data preprocessing steps involved handling missing values, feature standardization, and addressing class imbalance using SMOTE. To ensure explainability, SHAP plots were utilized to interpret the model’s predictions. The model achieved a high ROC-AUC score of 0.934 and F1-score of 0.851, outperforming baseline methods. Finally, a user-friendly application was deployed using Streamlit to enable practical usage of the prediction tool, demonstrating potential to alleviate healthcare burdens through interpretable and efficient prediction techniques.

## Dataset
The dataset used in this study is the Heart Disease UCI dataset from the UCI Repository (https://archive.ics.uci.edu/dataset/45/heart+disease), comprising 303 instances with 14 attributes that describe patient characteristics such as age, sex, and cholesterol levels.

<img width="383" alt="image" src="https://github.com/user-attachments/assets/695f4386-65ad-43c2-bed4-bd8e1ce2a9e2" />

To gain an initial understanding of the dataset, we computed descriptive statistics for all features. A summary of these descriptive statistics is presented as follow:

<img width="388" alt="image" src="https://github.com/user-attachments/assets/34d240bd-818d-4f1c-89ce-514ac93ba3cc" />

