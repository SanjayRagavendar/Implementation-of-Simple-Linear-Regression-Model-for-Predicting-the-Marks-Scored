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
```
```py
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

![307705140-77163267-9ac2-467b-803d-cbbb5d8682ae](https://github.com/SanjayRagavendar/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/91368803/291f6778-5468-43a2-a3fc-c0722297140c)

### Tail
![307705377-96c0a72b-a114-45ba-a43a-59e33a4b718e](https://github.com/SanjayRagavendar/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/91368803/e76ba7ca-ad5d-44df-942c-230f40d10835)


### X and Y values

![307722226-bf70d1b6-dddb-492a-b1ae-5788a4050c7d](https://github.com/SanjayRagavendar/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/91368803/6d186ca8-be59-4cb1-8ba0-f7c11a33c6cc)

![307722697-b725c8f6-a33d-4cd3-97c9-2f34b96115da](https://github.com/SanjayRagavendar/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/91368803/5dc0fea1-4a9c-43ae-af8a-ae590a765e8e)

### Prediction of X and Y

![307722795-2588d8af-ccd8-464c-adaf-eb18b67ba69f](https://github.com/SanjayRagavendar/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/91368803/fda853b7-d3b1-40dc-a2e0-1839d261f24a)

![307722827-41e9e8e8-170c-489a-995a-1c79e3556202](https://github.com/SanjayRagavendar/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/91368803/26f927fa-a4a7-4027-9213-79095d705497)

### MSS,MSE and RMSE 

![307722905-da31d005-1efa-4c44-b33b-6908d4be9350](https://github.com/SanjayRagavendar/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/91368803/98286727-2d1f-4aa0-b785-3f32638b841d)

### Training Set
![265499664-8719ffc5-5bc7-4a8e-9002-42f8be22da89](https://github.com/SanjayRagavendar/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/91368803/cd85f04a-85a0-4104-bad6-07cf03ab097f)

### Testing Set
![265499732-2afee5df-91f6-4421-b922-c7fa554885fd](https://github.com/SanjayRagavendar/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/91368803/7af30789-2b3c-4674-a0c1-10e3aca1205b)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
