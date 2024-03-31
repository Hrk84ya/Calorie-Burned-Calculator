#Project: Calorie Burned Calculator

#Importing the required modules
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns



#Extracting data from the datasets
df_exercise=pd.read_csv('exercise.csv')
df_calories=pd.read_csv('calories.csv')
df=pd.concat([df_exercise,df_calories['Calories']],axis=1)

le=LabelEncoder()
df['Gender']=le.fit_transform(df['Gender'])
X=df.drop(columns=['User_ID','Calories'])
y=df['Calories']

#Splitting the dataset into training and testing data sets
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=42)
model=XGBRegressor()
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
print(metrics.r2_score(y_pred,y_test))

#Predicting the Calories Burned by the User 
g=input("Enter Your sex: ")
gender=-1
if g.lower()=='male' or g.lower()=='m':
    gender=1
elif g.lower()=='female' or g.lower()=='f':
    gender=0
else:
    gender=-1

# Taking the input from the user with input validation
while True:
    try:
        age = int(input("Enter your age: "))
        if age <= 0:
            raise ValueError("Age must be a positive integer.")
        break
    except ValueError as e:
        print(e)

while True:
    try:
        height = float(input("Enter your Height (in cms): "))
        if height <= 0:
            raise ValueError("Height must be a positive value.")
        break
    except ValueError as e:
        print(e)

while True:
    try:
        weight = float(input("Enter your Weight (in kgs): "))
        if weight <= 0:
            raise ValueError("Weight must be a positive value.")
        break
    except ValueError as e:
        print(e)

while True:
    try:
        duration = float(input("Enter your Duration of Exercise (in mins): "))
        if duration < 0:
            raise ValueError("Duration cannot be negative.")
        break
    except ValueError as e:
        print(e)

while True:
    try:
        heart = float(input("Enter your Heart Rate after Exercise: "))
        break
    except ValueError as e:
        print(e)

while True:
    try:
        body_temp = float(input("Enter your Body Temperature (in celsius): "))
        break
    except ValueError as e:
        print(e)


try:
    # Predicting the Calories Burned by the User
    X_list = [gender, age, height, weight, duration, heart, body_temp]
    X_df = pd.DataFrame([X_list])
    X_df.columns = ['Gender', 'Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']
    y_pred_data = model.predict(X_df)
    if np.isnan(y_pred_data):
        raise ValueError("Prediction error: Model returned NaN.")
    print("Congratulations!!!\nYou Burnt %.3f CALORIES Today" % y_pred_data)
except Exception as e:
    print("Error:", e)


# Relationship between Duration of Exercise and Calories Burned
# Showing some general Insights
sns.set_style("whitegrid")

# Fit a polynomial curve
sns.regplot(x='Duration', y='Calories', data=df, order=2, ci=None, scatter_kws={'s': 10})

plt.xlabel('Duration (min)')
plt.ylabel('Calories Burned')
plt.title('Relationship between Duration of Exercise and Calories Burned')

plt.show()