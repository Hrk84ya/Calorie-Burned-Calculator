#Project: Enhanced Calorie Burned Calculator with Model Improvements

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Load and prepare data
df_exercise = pd.read_csv('exercise.csv')
df_calories = pd.read_csv('calories.csv')
df = pd.concat([df_exercise, df_calories['Calories']], axis=1)

# Feature Engineering
def create_features(df):
    df = df.copy()
    
    # BMI calculation
    df['BMI'] = df['Weight'] / ((df['Height'] / 100) ** 2)
    
    # Age groups
    df['Age_Group'] = pd.cut(df['Age'], bins=[0, 25, 35, 50, 100], labels=[0, 1, 2, 3])
    
    # Exercise intensity (based on heart rate and duration)
    df['Exercise_Intensity'] = (df['Heart_Rate'] * df['Duration']) / 1000
    
    # Heart rate zones (assuming max HR = 220 - age)
    df['HR_Zone'] = df['Heart_Rate'] / (220 - df['Age'])
    
    return df

df = create_features(df)

# Encode categorical variables
le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])
df['Age_Group'] = df['Age_Group'].astype(int)

# Prepare features and target
X = df.drop(columns=['User_ID', 'Calories'])
y = df['Calories']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Scale features for neural network
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model Comparison
models = {
    'XGBoost': XGBRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(random_state=42),
    'Linear Regression': LinearRegression(),
    'Neural Network': MLPRegressor(random_state=42, max_iter=1000)
}

print("=== Model Comparison ===")
model_results = {}

for name, model in models.items():
    if name == 'Neural Network':
        # Use scaled data for neural network
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
    
    r2 = metrics.r2_score(y_test, y_pred)
    mae = metrics.mean_absolute_error(y_test, y_pred)
    
    model_results[name] = {
        'R2': r2,
        'MAE': mae,
        'CV_Mean': cv_scores.mean(),
        'CV_Std': cv_scores.std()
    }
    
    print(f"{name}:")
    print(f"  R² Score: {r2:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  CV R² Mean: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
    print()

# Hyperparameter Tuning for best models
print("=== Hyperparameter Tuning ===")

# XGBoost tuning
xgb_params = {
    'n_estimators': [100, 200],
    'max_depth': [3, 6],
    'learning_rate': [0.1, 0.2]
}

xgb_grid = GridSearchCV(XGBRegressor(random_state=42), xgb_params, cv=3, scoring='r2')
xgb_grid.fit(X_train, y_train)

# Random Forest tuning
rf_params = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5]
}

rf_grid = GridSearchCV(RandomForestRegressor(random_state=42), rf_params, cv=3, scoring='r2')
rf_grid.fit(X_train, y_train)

print(f"Best XGBoost params: {xgb_grid.best_params_}")
print(f"Best XGBoost score: {xgb_grid.best_score_:.4f}")
print(f"Best Random Forest params: {rf_grid.best_params_}")
print(f"Best Random Forest score: {rf_grid.best_score_:.4f}")

# Use best model for predictions
best_model = xgb_grid.best_estimator_

# Feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': best_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n=== Feature Importance ===")
print(feature_importance)

# Visualization
plt.figure(figsize=(15, 10))

# Model comparison plot
plt.subplot(2, 2, 1)
model_names = list(model_results.keys())
r2_scores = [model_results[name]['R2'] for name in model_names]
plt.bar(model_names, r2_scores)
plt.title('Model R² Comparison')
plt.xticks(rotation=45)
plt.ylabel('R² Score')

# Feature importance plot
plt.subplot(2, 2, 2)
plt.barh(feature_importance['feature'][:8], feature_importance['importance'][:8])
plt.title('Top 8 Feature Importance')
plt.xlabel('Importance')

# Cross-validation scores
plt.subplot(2, 2, 3)
cv_means = [model_results[name]['CV_Mean'] for name in model_names]
cv_stds = [model_results[name]['CV_Std'] for name in model_names]
plt.errorbar(model_names, cv_means, yerr=cv_stds, fmt='o', capsize=5)
plt.title('Cross-Validation Scores')
plt.xticks(rotation=45)
plt.ylabel('CV R² Score')

# Prediction vs Actual
plt.subplot(2, 2, 4)
y_pred_best = best_model.predict(X_test)
plt.scatter(y_test, y_pred_best, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Calories')
plt.ylabel('Predicted Calories')
plt.title('Prediction vs Actual')

plt.tight_layout()
plt.show()

# Enhanced prediction function
def predict_calories_enhanced(gender, age, height, weight, duration, heart_rate, body_temp):
    # Create feature vector with engineered features
    bmi = weight / ((height / 100) ** 2)
    age_group = 0 if age <= 25 else 1 if age <= 35 else 2 if age <= 50 else 3
    exercise_intensity = (heart_rate * duration) / 1000
    hr_zone = heart_rate / (220 - age)
    
    gender_encoded = 1 if gender.lower() in ['male', 'm'] else 0
    
    features = pd.DataFrame([[
        gender_encoded, age, height, weight, duration, heart_rate, body_temp,
        bmi, age_group, exercise_intensity, hr_zone
    ]], columns=X.columns)
    
    prediction = best_model.predict(features)[0]
    return prediction

# Interactive prediction
print("\n=== Enhanced Calorie Prediction ===")
try:
    g = input("Enter Your sex: ")
    age = int(input("Enter your age: "))
    height = float(input("Enter your Height (in cms): "))
    weight = float(input("Enter your Weight (in kgs): "))
    duration = float(input("Enter your Duration of Exercise (in mins): "))
    heart = float(input("Enter your Heart Rate after Exercise: "))
    body_temp = float(input("Enter your Body Temperature (in celsius): "))
    
    predicted_calories = predict_calories_enhanced(g, age, height, weight, duration, heart, body_temp)
    print(f"\nEnhanced Prediction: You burnt {predicted_calories:.1f} CALORIES!")
    
    # Calculate BMI and provide additional insights
    bmi = weight / ((height / 100) ** 2)
    print(f"Your BMI: {bmi:.1f}")
    print(f"Exercise Intensity Score: {(heart * duration) / 1000:.2f}")
    
except Exception as e:
    print(f"Error: {e}")