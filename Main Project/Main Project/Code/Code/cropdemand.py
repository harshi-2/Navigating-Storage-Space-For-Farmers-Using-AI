import pandas as pd
import numpy as np

df=pd.read_csv("Date-Wise-Prices-all-Commodity.csv")
df.head()

duplicated_count=df.duplicated().sum()
print("Number of duplicate entries: ", duplicated_count)

null_count=df.isnull().sum()
print("Number of null entries: ", null_count)

#Data Preprocessing

#Unique Market Locations of AgriFarm
df["Market"].unique()

#Convert Categorical Data of to Numerical Data to apply ML Algorithms
from sklearn.preprocessing import LabelEncoder
label_encoder=LabelEncoder()

df["Market"]=label_encoder.fit_transform(df["Market"])
df["Commodity"]=label_encoder.fit_transform(df["Commodity"])
df["Grade"]=label_encoder.fit_transform(df["Grade"])
df["Variety"]=label_encoder.fit_transform(df["Variety"])

#Dataset with Numerical Data
df.head()

#Representation of Data Graphically: HeatMap

#Selecting Features for Predicting Product Price
data_X=df.iloc[:,[4,5,6,7,9,10]].values
Y=df.iloc[:,-1].values

#Standard Scaler: Remove the mean and scales each feature/variable to unit variance
from sklearn.preprocessing import StandardScaler
ss=StandardScaler()
data_X=ss.fit_transform(data_X)

#Splitting the Data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(data_X,Y,test_size=0.2,random_state=42)
import math
#Applying Linear Regression Model to our dataset: Makes predictions for numeric variables
#80% Data is taken for training
from sklearn.linear_model import LinearRegression 
linear_reg=LinearRegression()
linear_reg.fit(x_train,y_train)
y_predict=linear_reg.predict(x_test)


mse = []
rmse = []
r2score = []
print("=================Linear Regression=================")    
#r2 Score: (total variance explained by model) / total variance.
from sklearn.metrics import mean_squared_error, r2_score
print("mse:", end= " ")
print(mean_squared_error(y_test,y_predict))
print("rmse:", end= " ")
print(math.sqrt(mean_squared_error(y_test,y_predict)))
print("r2 Score:", end= " ")
print(r2_score(y_test,y_predict)*100)
print("==================================================")    

mse.append(mean_squared_error(y_test,y_predict))
rmse.append(math.sqrt(mean_squared_error(y_test,y_predict)))
r2score.append(r2_score(y_test,y_predict)*100)
#Applying Ridge Regression Model to our dataset
from sklearn.linear_model import Ridge
ridge_reg=Ridge(alpha=0.001)
ridge_reg.fit(x_train,y_train)
y_predict2=ridge_reg.predict(x_test)

print("=================Ridge Regression=================")    
#r2 Score: (total variance explained by model) / total variance.
from sklearn.metrics import mean_squared_error, r2_score
print("mse:", end= " ")
print(mean_squared_error(y_test,y_predict2))
print("rmse:", end= " ")
print(math.sqrt(mean_squared_error(y_test,y_predict2)))
print("r2 Score:", end= " ")
print(r2_score(y_test,y_predict2)*100)
print("==================================================") 

mse.append(mean_squared_error(y_test,y_predict2))
rmse.append(math.sqrt(mean_squared_error(y_test,y_predict2)))
r2score.append(r2_score(y_test,y_predict2)*100)
#Applying Ridge Regression Model to our dataset
from sklearn.linear_model import Lasso
lasso_reg=Lasso(alpha=1.144)
lasso_reg.fit(x_train,y_train)
y_predict3=lasso_reg.predict(x_test)
print("=================Lasso Regression=================")   

#r2 Score: (total variance explained by model) / total variance.
from sklearn.metrics import mean_squared_error, r2_score
print("mse:", end= " ")
print(mean_squared_error(y_test,y_predict3))
print("rmse:", end= " ")
print(math.sqrt(mean_squared_error(y_test,y_predict3)))
print("r2 Score:", end= " ")
print(r2_score(y_test,y_predict3)*100)
print("==================================================") 

mse.append(mean_squared_error(y_test,y_predict3))
rmse.append(math.sqrt(mean_squared_error(y_test,y_predict3)))
r2score.append(r2_score(y_test,y_predict3)*100)
#As the r2 score percentage of Lasso Regression is maximum, so we will use this model for deployment of our project
import pickle
filename = 'best_model.sav'
pickle.dump(lasso_reg, open(filename, 'wb'))

alg = ['Linear Regression', 'Ridge Refgression', 'Lasso Regression']


resultmse = open('resmse.csv', 'w')
resultmse.write("algorithm,mse"+"\n")
for i in range(0, len(mse)):
    resultmse.write(alg[i]+","+str(mse[i])+"\n")
resultmse.close()



resultmse = open('resrmse.csv', 'w')
resultmse.write("algorithm,rmse"+"\n")
for i in range(0, len(rmse)):
    resultmse.write(alg[i]+","+str(rmse[i])+"\n")
resultmse.close()


resultr2score = open('resr2score.csv', 'w')
resultr2score.write("algorithm,r2score"+"\n")
for i in range(0, len(r2score)):
    resultr2score.write(alg[i]+","+str(r2score[i])+"\n")
resultr2score.close()

import matplotlib.pyplot as plt


fig = plt.figure(0)
df1 = pd.read_csv("resmse.csv")
al = df1['algorithm']
msevalue = df1['mse']
plt.bar(al, msevalue, align = 'center')
plt.xlabel("Algorithms ")
plt.ylabel("MSE Value")
plt.show()
fig.savefig('msegraph.jpg')
plt.close()

fig = plt.figure(1)
df2 = pd.read_csv("resrmse.csv")
al2 = df2['algorithm']
accvalue = df2['rmse']
plt.bar(al2, accvalue, align = 'center')
plt.xlabel("Algorithms ")
plt.ylabel("RMSE Value")
plt.show()
fig.savefig('rmsegraph.jpg')
plt.close()


fig = plt.figure(2)
df3 = pd.read_csv("resr2score.csv")
al3 = df3['algorithm']
accvalue = df3['r2score']
plt.bar(al3, accvalue, align = 'center')
plt.xlabel("Algorithms ")
plt.ylabel("R2Score Value")
plt.show()
fig.savefig('r2scoregraph.jpg')
plt.close()