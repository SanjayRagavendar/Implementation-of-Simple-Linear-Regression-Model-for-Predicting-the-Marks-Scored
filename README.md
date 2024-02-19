# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. **Import Libraries**: Import pandas, numpy, matplotlib.pyplot, and LinearRegression from scikit-learn.
2. **Read Data**: Read 'student_scores.csv' into a DataFrame (df) using pd.read_csv().
3. **Data Preparation**: Extract 'Hours' (x) and 'Scores' (y). Split data using train_test_split().
4. **Model Training**: Create regressor instance. Fit model with regressor.fit(x_train, y_train).
5. **Prediction**: Predict scores (y_pred) using regressor.predict(x_test).
6. **Model Evaluation & Visualization**: Calculate errors. Plot training and testing data. Print errors.

## Program:
```py
#Program to implement the simple linear regression model for predicting the marks scored.
#Developed by: Sanjay Ragavendar M K
#RegisterNumber:  212222100045

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error

df=pd.read_csv("/content/student_scores.csv")
print(df.head())

print(df.tail())

x = df.iloc[:,:-1].values
print(x)

y = df.iloc[:,1].values
print(y)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
print(y_pred)
print(y_test)

#Graph plot for training data
plt.scatter(x_train,y_train,color='orange')
plt.plot(x_train,regressor.predict(x_train),color='red')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

#Graph plot for test data
plt.scatter(x_test,y_test,color='purple')
plt.plot(x_train,regressor.predict(x_train),color='yellow')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse) 
```

## Output:
### Head
![265497944-240b2352-15b4-43d4-a1e9-5bf7cff9c73c](https://github.com/SanjayRagavendar/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/91368803/c12f0ba7-7e54-49f8-bbab-d4798be56610)

### Tail
![265498034-74f38c0c-076a-4468-b7ec-beda04215aa7](https://github.com/SanjayRagavendar/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/91368803/3b4351f4-8ac0-4eaf-9b30-09ce33dcc5a0)

### X and Y values
![265498293-93dc266d-8193-46b9-aacb-f235729cd7eb](https://github.com/SanjayRagavendar/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/91368803/24181ef0-f820-4628-83a9-6f38dd3f4a07)

### Prediction of X and Y
![265498417-1c6dbe39-5ea4-4fad-9f86-4115bcb5f5b6](https://github.com/SanjayRagavendar/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/91368803/d516f109-5227-4fd1-9f6c-ae023db135a2)

### MSS,MSE and RMSE 
![265498528-7bbe9578-3bc9-4420-8f78-1c84e39fe21f](https://github.com/SanjayRagavendar/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/91368803/1f372d2e-e4ae-4c7a-8926-5dc9ab3a8cdb)

### Training Set
![265499664-8719ffc5-5bc7-4a8e-9002-42f8be22da89](https://github.com/SanjayRagavendar/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/91368803/cd85f04a-85a0-4104-bad6-07cf03ab097f)

### Testing Set
![265499732-2afee5df-91f6-4421-b922-c7fa554885fd](https://github.com/SanjayRagavendar/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/91368803/7af30789-2b3c-4674-a0c1-10e3aca1205b)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
