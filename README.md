 # Project: Enhanced Calorie Burned Calculator

### Overview
This project predicts calories burned during exercise using advanced machine learning techniques. It features model comparison, hyperparameter tuning, cross-validation, and enhanced feature engineering for superior accuracy and insights.

### Files Included
1. `exercise.csv`: Dataset containing exercise-related information
2. `calories.csv`: Dataset providing information on calories burned
3. `Calories.py`: Basic calorie prediction script
4. `enhanced_calories.py`: **Advanced version with model improvements**
5. `project.ipynb`: Jupyter notebook for data visualization
6. `requirements.txt`: Required Python packages

### Installation
```bash
pip install -r requirements.txt
```

### Usage

#### Basic Version
```bash
python Calories.py
```

#### Enhanced Version (Recommended)
```bash
python enhanced_calories.py
```

### Enhanced Features

#### Model Improvements
- **Model Comparison**: Tests XGBoost, Random Forest, Linear Regression, and Neural Networks
- **Hyperparameter Tuning**: Automatic optimization using GridSearchCV
- **Cross-Validation**: 5-fold validation for robust performance assessment
- **Feature Engineering**: BMI, age groups, exercise intensity, and heart rate zones

#### Advanced Analytics
- **Feature Importance Analysis**: Shows which factors most influence calorie burn
- **Model Performance Metrics**: R², MAE, and cross-validation scores
- **Enhanced Visualizations**: Model comparison charts and feature importance plots
- **Additional Health Insights**: BMI calculation and exercise intensity scoring

### Input Parameters
- Gender (Male/Female)
- Age (years)
- Height (cm)
- Weight (kg)
- Duration of Exercise (mins)
- Heart Rate after Exercise
- Body Temperature (°C)

### Results
The enhanced version provides:
- **Accurate calorie predictions** using the best-performing model
- **Model comparison results** with performance metrics
- **Feature importance rankings** showing key factors
- **Additional health metrics** (BMI, exercise intensity)
- **Comprehensive visualizations** for data insights

### Technical Details
- **Algorithms**: XGBoost, Random Forest, Linear Regression, Neural Networks
- **Validation**: 5-fold cross-validation with hyperparameter tuning
- **Features**: Original + engineered features (BMI, age groups, intensity scores)
- **Metrics**: R² score, MAE, cross-validation performance

### Performance
The enhanced model typically achieves:
- R² scores > 0.95 on test data
- Robust cross-validation performance
- Automatic selection of optimal hyperparameters

For best results, use the enhanced version which provides superior accuracy and comprehensive health insights!
