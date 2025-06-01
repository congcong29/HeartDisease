# Import necessary libraries
# Import necessary libraries
import os  # 新增导入 os
import tempfile  # 新增导入 tempfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import shap
import warnings
import joblib  # joblib 必须在这里导入
warnings.filterwarnings('ignore')

# 设置 joblib 临时目录到系统的临时文件夹，避免非 ASCII 路径问题
joblib_temp_dir = tempfile.gettempdir()
os.environ['JOBLIB_TEMP_FOLDER'] = joblib_temp_dir
class HeartDiseaseAnalysis:
    def __init__(self, data_url):
        self.columns = [
            "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
            "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"
        ]
        self.data_url = data_url
        self.load_data()

    def load_data(self):
        """Load and prepare the dataset"""
        self.df = pd.read_csv(self.data_url, names=self.columns, na_values="?")
        self.X = self.df.drop('target', axis=1)
        self.y = (self.df['target'] > 0).astype(int)

        # Display dataset metadata
        print("Heart Disease dataset from UCI ML Repository containing patient information and heart disease presence.")
        print(self.df.describe())

    def preprocess_data(self):
        """Handle missing values, standardize features, and handle imbalance"""
        # Handle missing values
        self.X = self.X.copy()
        for col in self.X.columns:
            if self.X[col].dtype in ['int64', 'float64']:
                self.X[col].fillna(self.X[col].median(), inplace=True)
            else:
                self.X[col].fillna(self.X[col].mode()[0], inplace=True)

        # Feature Engineering
        self.create_new_features()

        # Standardizing numerical features
        self.scaler = StandardScaler()
        self.X_scaled = self.scaler.fit_transform(self.X)

        # Handle class imbalance using SMOTE
        smote = SMOTE(random_state=42)
        self.X_resampled, self.y_resampled = smote.fit_resample(self.X_scaled, self.y)

        # Split the dataset
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X_resampled, self.y_resampled, test_size=0.3, random_state=42)

    def create_new_features(self):
        """Create new features based on domain knowledge"""
        self.X['age_chol_interaction'] = self.X['age'] * self.X['chol']
        self.X['thalach_trestbps_ratio'] = self.X['thalach'] / (self.X['trestbps'] + 1)
        self.X['age_bp_ratio'] = self.X['age'] / (self.X['trestbps'] + 1)
        self.X['heart_reserve'] = 220 - self.X['age'] - self.X['thalach']

    def perform_eda(self):
        """Perform Exploratory Data Analysis"""
        # Correlation analysis
        plt.figure(figsize=(12, 8))
        correlation_matrix = self.X.corr()
        mask = np.zeros_like(correlation_matrix)
        mask[np.triu_indices_from(mask)] = True
        sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', center=0)
        plt.title('Feature Correlation Heatmap')
        plt.tight_layout()
        plt.show()

        # Feature distribution analysis
        numerical_features = ["age", "trestbps", "chol", "thalach", "oldpeak"]
        for feature in numerical_features:
            plt.figure(figsize=(6, 4))
            sns.histplot(self.df[feature], kde=True)
            plt.title(f'Distribution of {feature}')
            plt.xlabel(feature)
            plt.ylabel('Frequency')
            plt.show()

        # Categorical feature distribution by heart disease presence
        categorical_features = ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"]
        for feature in categorical_features:
            plt.figure(figsize=(6, 4))
            sns.countplot(x=feature, hue=self.y, data=self.df)
            plt.title(f'{feature} Distribution by Heart Disease Presence')
            plt.xlabel(feature)
            plt.ylabel('Count')
            plt.legend(title='Heart Disease', loc='upper right', labels=['No', 'Yes'])
            plt.show()

        # Reduced Pair Plot for Selected Features
        selected_features = ["age", "trestbps", "chol", "thalach", "oldpeak"]
        sns.pairplot(self.df[selected_features])
        plt.show()

        # Box plots for numerical features by target variable
        for feature in numerical_features:
            plt.figure(figsize=(8, 6))
            sns.boxplot(x=self.y, y=self.df[feature])
            plt.title(f'{feature} distribution by target variable')
            plt.xlabel('Heart Disease (0 = No, 1 = Yes)')
            plt.ylabel(feature)
            plt.show()

    def train_model(self):
        """Train and tune XGBoost model"""
        # Initial model
        self.model = xgb.XGBClassifier(
            objective='binary:logistic',
            use_label_encoder=False,
            eval_metric='logloss'
        )

        # Cross-validation
        cv_scores = cross_val_score(self.model, self.X_train, self.y_train, cv=5)
        print(f"Cross-validation scores: {cv_scores}")
        print(f"Average CV score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")

        # Hyperparameter tuning using GridSearchCV
        param_grid = {
            'max_depth': [3, 4, 5, 6],
            'learning_rate': [0.01, 0.05, 0.1],
            'n_estimators': [100, 200, 300],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0]
        }

        grid_search = GridSearchCV(
            estimator=self.model,
            param_grid=param_grid,
            cv=5,
            scoring='roc_auc',
            n_jobs=-1,
            verbose=1
        )
        # 设置 joblib 临时目录到系统的临时文件夹，避免非 ASCII 路径问题
        joblib_temp_dir = tempfile.gettempdir()
        os.environ['JOBLIB_TEMP_FOLDER'] = joblib_temp_dir

        grid_search.fit(self.X_train, self.y_train)
        self.best_params = grid_search.best_params_
        print(f"Best parameters found: {self.best_params}")
        print(f"Best cross-validation score: {grid_search.best_score_:.3f}")

        # Training final model
        self.final_model = xgb.XGBClassifier(**self.best_params,
                                           use_label_encoder=False,
                                           eval_metric='logloss')
        self.final_model.fit(self.X_train, self.y_train)

    def evaluate_model(self):
        """Evaluate model performance"""
        y_pred = self.final_model.predict(self.X_test)
        y_prob = self.final_model.predict_proba(self.X_test)[:, 1]

        # Calculate and display metrics
        metrics = {
            'Accuracy': accuracy_score(self.y_test, y_pred),
            'Precision': precision_score(self.y_test, y_pred),
            'Recall': recall_score(self.y_test, y_pred),
            'F1': f1_score(self.y_test, y_pred)
        }

        for metric, value in metrics.items():
            print(f"{metric}: {value:.3f}")

        # ROC Curve
        fpr, tpr, _ = roc_curve(self.y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(10, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:0.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc='lower right')
        plt.show()

        # Confusion Matrix
        conf_matrix = confusion_matrix(self.y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix Heatmap')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.show()

    def interpret_model(self):
        """Model interpretation using SHAP values"""
        explainer = shap.Explainer(self.final_model, self.X_train)
        shap_values = explainer(self.X_test)

        # SHAP summary plot for overall feature importance
        shap.summary_plot(shap_values, self.X_test,
                         feature_names=self.X.columns,
                         plot_type="bar")

        # Detailed explanation for individual predictions using SHAP force plot
        shap.initjs()
        for i in range(3):  # Display for first 3 test instances
            shap.force_plot(explainer.expected_value, shap_values[i].values, self.X_test[i], feature_names=self.X.columns)

# Usage
if __name__ == "__main__":
    print(tempfile.gettempdir())

    data_url = "processed.cleveland.data"

    # Initialize analyzer
    analyzer = HeartDiseaseAnalysis(data_url)

    # Data preprocessing
    analyzer.preprocess_data()

    # Perform Exploratory Data Analysis (EDA)
    # analyzer.perform_eda()

    # Train the model
    analyzer.train_model()

    # Evaluate the model
    analyzer.evaluate_model()

    # Interpret the model
    analyzer.interpret_model()

    # Save the model
    joblib.dump(analyzer.final_model, 'XGBoost.pkl')
    print("Model saved as 'XGBoost.pkl'")