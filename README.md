 #Project: Calorie Burned Calculator

### Overview
This project aims to predict the calories burned by a user during exercise based on various input parameters. It utilizes machine learning techniques and XGBoost regression to make accurate predictions. The user can input personal information and exercise details to calculate the calories burned.

### Files Included
1. `exercise.csv`: Dataset containing exercise-related information.
2. `calories.csv`: Dataset providing information on calories burned.
3. `Calories.py`: Python script to predict calories burned and visualize insights.

### Usage
1. Ensure the required modules are installed (`pandas`, `numpy`, `scikit-learn`, `xgboost`, `matplotlib`).
2. Run the `Calories.py` script to predict calories burned and visualize exercise insights.

### Instructions
1. Run the script and input the required data:
   - Gender (Male/Female)
   - Age (years)
   - Height (cm)
   - Weight (kg)
   - Duration of Exercise (mins)
   - Heart Rate after Exercise
   - Body Temperature (°C)

2. The script will predict the calories burned by the user based on the provided information.

### Results
Upon execution, the script will display:
- The predicted calories burned by the user.
- A scatter plot showing the relationship between exercise duration and calories burned.

### Note
- Make sure to input accurate and valid data for precise calorie burn prediction.
- The model's accuracy can be assessed using the R-squared score calculation.

For any issues or further information, please refer to the script and datasets included. Enjoy understanding your calorie burn dynamics!  
