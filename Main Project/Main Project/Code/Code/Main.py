import tkinter as tk
from tkinter import Message ,Text
#import cv2,os
import shutil
import csv
import numpy as np
from PIL import Image, ImageTk
import pandas as pd
import datetime
import time
import tkinter.ttk as ttk
import tkinter.font as font
import os
#import logistic as log
#import first as f
#import second as s
#import plograph as ac
from tkinter import *
from matplotlib import pyplot as plt

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score
from_date = datetime.datetime.today()
from tkinter import filedialog

currentDate = time.strftime("%d_%m_%y")

#font = cv2.FONT_HERSHEY_SIMPLEX
#fontScale=1
#fontColor=(255,255,255)

cond=0


window = tk.Tk()
window.title("CropPrice Freeze")


window.geometry('1280x720')
window.configure(background='gray')

#window.attributes('-fullscreen', True)

window.grid_rowconfigure(0, weight=1)
window.grid_columnconfigure(0, weight=1)


message1 = tk.Label(window, text="CropPrice Freeze" ,bg="light blue"  ,fg="white"  ,width=50  ,height=1, font=('times', 30, 'italic bold underline')) 
message1.place(x=80, y=20)

dataselect = StringVar()
dataselect.set(None)


outputvar = StringVar()

list1= ['Linear Regression','Ridge Regression','Lasso Regression']	

mse = []
rmse = []    
r2score = []

def Lasso():
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
    data_X=df.iloc[:,[4,5,6,7]].values
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
    
    #Applying Lasso Regression Model to our dataset
    from sklearn.linear_model import Ridge
    ridge_reg=Ridge(alpha=0.001)
    ridge_reg.fit(x_train,y_train)
    y_predict2=lasso_reg.predict(x_test)
    
    
    
    x1 = dataentry.get()
    
    print(x1)
    
    
    

    
        
    df1 = pd.read_csv(x1)
    
    dataf = pd.read_csv(x1)
    
    df1.head()

    duplicated_count=df1.duplicated().sum()
    print("Number of duplicate entries: ", duplicated_count)

    null_count=df1.isnull().sum()
    print("Number of null entries: ", null_count)

    #Data Preprocessing

    #Unique Market Locations of AgriFarm
    df1["Market"].unique()

    #Convert Categorical Data of to Numerical Data to apply ML Algorithms
    from sklearn.preprocessing import LabelEncoder
    label_encoder=LabelEncoder()

    df1["Market"]=label_encoder.fit_transform(df1["Market"])
    
    
    if df1["Commodity"].any()=="Apple":
        df1["Commodity"] = 0
    elif df1["Commodity"].any()=="Banana":
        df1["Commodity"] = 1
    elif df1["Commodity"].any()=="Cabbage":
        df1["Commodity"] = 2
    elif df1["Commodity"].any()=="Gur(Jaggery)":
        df1["Commodity"] = 3
    elif df1["Commodity"].any()=="Lemon":
        df1["Commodity"] = 4
    elif df1["Commodity"].any()=="Maize":
        df1["Commodity"] = 5
    elif df1["Commodity"].any()=="Onion":
        df1["Commodity"] = 6
    elif df1["Commodity"].any()=="Potato":
        df1["Commodity"] = 7
    else:
        df1["Commodity"] = 8
        
        
    #df1["Commodity"]=label_encoder.fit_transform(df1["Commodity"])
    
    
    
    
    if df1["Grade"].any()=="Hybrid":
        df1["Grade"] = 0
    elif df1["Grade"].any()=="Local":
        df1["Grade"] = 1
    elif df1["Grade"].any()=="Other":
        df1["Grade"] = 2
    elif df1["Grade"].any()=="NO 1":
        df1["Grade"] = 3
    elif df1["Grade"].any()=="NO 2":
        df1["Grade"] = 4
    else:
        df1["Grade"] = 5
    
    
    #df1["Grade"]=label_encoder.fit_transform(df1["Grade"])
    
    if df1["Variety"].any()=="FAQ":
        df1["Variety"] = 0
    else:
        df1["Variety"] = 1
    
    #df1["Variety"]=label_encoder.fit_transform(df1["Variety"])

    #Dataset with Numerical Data
    df1.head()

    #Representation of Data Graphically: HeatMap

    #Selecting Features for Predicting Product Price
    data_X=df1.iloc[:,[4,5,6,7]].values
   
    #dftest = df1[:30][['DEWP','TEMP','PRES','Iws','Is','Ir']]
    
    #Xtest =dftest[['DEWP','TEMP','PRES','Iws','Is','Ir']]
    
    y_pred = lasso_reg.predict(data_X)
    
    #inp=[-16,-4.0,1020.0,1.79,0,0]
    #y_pred = clf.predict([inp])
    print("Predicted Values for Lasso Regression Input: ")
    print(y_pred)
    #outputvar.set(y_pred[0][0])
    
    text.insert(END,"======Lasso Regression======"+"\n\n")      
    text.insert(END,"Y=%s" %(y_pred)+"\n\n")    
    text.insert(END,"Y=%s" %(dataf["Commodity"])+"\n\n")
    text.insert(END,"If the Given Crop in Demand(Put it for Freeze) according to below conditions:"+"\n\n")
    text.insert(END,"1. Gur (Jaggery ) > 3000"+"\n\n")
    text.insert(END,"2. Tomato > 1000"+"\n\n")
    text.insert(END,"3. Onion > 1000"+"\n\n")
    text.insert(END,"4. Maize > 1000"+"\n\n")
    text.insert(END,"5. Potato > 1000"+"\n\n")
    text.insert(END,"6. Banana > 1000"+"\n\n")
    text.insert(END,"7. Apple > 1000"+"\n\n")
    text.insert(END,"8. Cabbage > 1000"+"\n\n")
    text.insert(END,"9. Lemon > 1000"+"\n\n")
    

def Ridge():
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
    data_X=df.iloc[:,[4,5,6,7]].values
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
    from sklearn.linear_model import Ridge
    ridge_reg=Ridge(alpha=0.001)
    ridge_reg.fit(x_train,y_train)
    y_predict2=ridge_reg.predict(x_test)
    
    
    
    x1 = dataentry.get()
    
    print(x1)
    
    dataf = pd.read_csv(x1)
    
        
    df1 = pd.read_csv(x1)
    
    df1.head()

    duplicated_count=df1.duplicated().sum()
    print("Number of duplicate entries: ", duplicated_count)

    null_count=df1.isnull().sum()
    print("Number of null entries: ", null_count)

    #Data Preprocessing

    #Unique Market Locations of AgriFarm
    df1["Market"].unique()

    #Convert Categorical Data of to Numerical Data to apply ML Algorithms
    from sklearn.preprocessing import LabelEncoder
    label_encoder=LabelEncoder()

    df1["Market"]=label_encoder.fit_transform(df1["Market"])
    
    
    if df1["Commodity"].any()=="Apple":
        df1["Commodity"] = 0
    elif df1["Commodity"].any()=="Banana":
        df1["Commodity"] = 1
    elif df1["Commodity"].any()=="Cabbage":
        df1["Commodity"] = 2
    elif df1["Commodity"].any()=="Gur(Jaggery)":
        df1["Commodity"] = 3
    elif df1["Commodity"].any()=="Lemon":
        df1["Commodity"] = 4
    elif df1["Commodity"].any()=="Maize":
        df1["Commodity"] = 5
    elif df1["Commodity"].any()=="Onion":
        df1["Commodity"] = 6
    elif df1["Commodity"].any()=="Potato":
        df1["Commodity"] = 7
    else:
        df1["Commodity"] = 8
        
        
    #df1["Commodity"]=label_encoder.fit_transform(df1["Commodity"])
    
    
    
    
    if df1["Grade"].any()=="Hybrid":
        df1["Grade"] = 0
    elif df1["Grade"].any()=="Local":
        df1["Grade"] = 1
    elif df1["Grade"].any()=="Other":
        df1["Grade"] = 2
    elif df1["Grade"].any()=="NO 1":
        df1["Grade"] = 3
    elif df1["Grade"].any()=="NO 2":
        df1["Grade"] = 4
    else:
        df1["Grade"] = 5
    
    
    #df1["Grade"]=label_encoder.fit_transform(df1["Grade"])
    
    if df1["Variety"].any()=="FAQ":
        df1["Variety"] = 0
    else:
        df1["Variety"] = 1
    
    #df1["Variety"]=label_encoder.fit_transform(df1["Variety"])

    #Dataset with Numerical Data
    df1.head()

    #Representation of Data Graphically: HeatMap

    #Selecting Features for Predicting Product Price
    data_X=df1.iloc[:,[4,5,6,7]].values
   
    #dftest = df1[:30][['DEWP','TEMP','PRES','Iws','Is','Ir']]
    
    #Xtest =dftest[['DEWP','TEMP','PRES','Iws','Is','Ir']]
    
    y_pred = ridge_reg.predict(data_X)
    
    #inp=[-16,-4.0,1020.0,1.79,0,0]
    #y_pred = clf.predict([inp])
    print("Predicted Values for Ridge Regression Input: ")
    print(y_pred)
    #outputvar.set(y_pred[0][0])
    
    text.insert(END,"======Ridge Regression======"+"\n\n")      
    text.insert(END,"Y=%s" %(y_pred)+"\n\n")
    text.insert(END,"Y=%s" %(dataf["Commodity"])+"\n\n")
    text.insert(END,"If the Given Crop in Demand(Put it for Freeze) according to below conditions:"+"\n\n")
    text.insert(END,"1. Gur (Jaggery ) > 3000"+"\n\n")
    text.insert(END,"2. Tomato > 1000"+"\n\n")
    text.insert(END,"3. Onion > 1000"+"\n\n")
    text.insert(END,"4. Maize > 1000"+"\n\n")
    text.insert(END,"5. Potato > 1000"+"\n\n")
    text.insert(END,"6. Banana > 1000"+"\n\n")
    text.insert(END,"7. Apple > 1000"+"\n\n")
    text.insert(END,"8. Cabbage > 1000"+"\n\n")
    text.insert(END,"9. Lemon > 1000"+"\n\n")
    
    
    
    

def LR():
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
    data_X=df.iloc[:,[4,5,6,7]].values
    Y=df.iloc[:,-1].values

    #Standard Scaler: Remove the mean and scales each feature/variable to unit variance
    from sklearn.preprocessing import StandardScaler
    ss=StandardScaler()
    data_X=ss.fit_transform(data_X)

    #Splitting the Data
    from sklearn.model_selection import train_test_split
    x_train,x_test,y_train,y_test=train_test_split(data_X,Y,test_size=0.2,random_state=42)
    import math   
    
    
    
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
    
    
    #Applying Linear Regression Model to our dataset
    from sklearn.linear_model import Ridge
    ridge_reg=Ridge(alpha=0.001)
    ridge_reg.fit(x_train,y_train)
    y_predict2=ridge_reg.predict(x_test)
    
    
    
    x1 = dataentry.get()
    
    print(x1)
    
    
    
        
    df1 = pd.read_csv(x1)
    
    dataf = pd.read_csv(x1)
    
    df1.head()

    duplicated_count=df1.duplicated().sum()
    print("Number of duplicate entries: ", duplicated_count)

    null_count=df1.isnull().sum()
    print("Number of null entries: ", null_count)

    #Data Preprocessing

    #Unique Market Locations of AgriFarm
    df1["Market"].unique()

    #Convert Categorical Data of to Numerical Data to apply ML Algorithms
    from sklearn.preprocessing import LabelEncoder
    label_encoder=LabelEncoder()

    df1["Market"]=label_encoder.fit_transform(df1["Market"])
    
    
    if df1["Commodity"].any()=="Apple":
        df1["Commodity"] = 0
    elif df1["Commodity"].any()=="Banana":
        df1["Commodity"] = 1
    elif df1["Commodity"].any()=="Cabbage":
        df1["Commodity"] = 2
    elif df1["Commodity"].any()=="Gur(Jaggery)":
        df1["Commodity"] = 3
    elif df1["Commodity"].any()=="Lemon":
        df1["Commodity"] = 4
    elif df1["Commodity"].any()=="Maize":
        df1["Commodity"] = 5
    elif df1["Commodity"].any()=="Onion":
        df1["Commodity"] = 6
    elif df1["Commodity"].any()=="Potato":
        df1["Commodity"] = 7
    else:
        df1["Commodity"] = 8
        
        
    #df1["Commodity"]=label_encoder.fit_transform(df1["Commodity"])
    
    
    
    
    if df1["Grade"].any()=="Hybrid":
        df1["Grade"] = 0
    elif df1["Grade"].any()=="Local":
        df1["Grade"] = 1
    elif df1["Grade"].any()=="Other":
        df1["Grade"] = 2
    elif df1["Grade"].any()=="NO 1":
        df1["Grade"] = 3
    elif df1["Grade"].any()=="NO 2":
        df1["Grade"] = 4
    else:
        df1["Grade"] = 5
    
    
    #df1["Grade"]=label_encoder.fit_transform(df1["Grade"])
    
    if df1["Variety"].any()=="FAQ":
        df1["Variety"] = 0
    else:
        df1["Variety"] = 1
    
    #df1["Variety"]=label_encoder.fit_transform(df1["Variety"])

    #Dataset with Numerical Data
    df1.head()

    #Representation of Data Graphically: HeatMap

    #Selecting Features for Predicting Product Price
    data_X=df1.iloc[:,[4,5,6,7]].values
   
    #dftest = df1[:30][['DEWP','TEMP','PRES','Iws','Is','Ir']]
    
    #Xtest =dftest[['DEWP','TEMP','PRES','Iws','Is','Ir']]
    
    y_pred = linear_reg.predict(data_X)
    
    #inp=[-16,-4.0,1020.0,1.79,0,0]
    #y_pred = clf.predict([inp])
    print("Predicted Values for Linear Regression Input: ")
    print(y_pred)
    #outputvar.set(y_pred[0][0])
    
    text.insert(END,"======Linear Regression======"+"\n\n")      
    text.insert(END,"Y=%s" %(y_pred)+"\n\n")
    text.insert(END,"Y=%s" %(dataf["Commodity"])+"\n\n")
    text.insert(END,"If the Given Crop in Demand(Put it for Freeze) according to below conditions:"+"\n\n")
    text.insert(END,"1. Gur (Jaggery ) > 3000"+"\n\n")
    text.insert(END,"2. Tomato > 1000"+"\n\n")
    text.insert(END,"3. Onion > 1000"+"\n\n")
    text.insert(END,"4. Maize > 1000"+"\n\n")
    text.insert(END,"5. Potato > 1000"+"\n\n")
    text.insert(END,"6. Banana > 1000"+"\n\n")
    text.insert(END,"7. Apple > 1000"+"\n\n")
    text.insert(END,"8. Cabbage > 1000"+"\n\n")
    text.insert(END,"9. Lemon > 1000"+"\n\n")

def predict():
    
    ds = dataselect.get()
    print(ds)
    if ds == 'Linear Regression':
        print('go to Linear Regression Algorithm')
        LR()
    elif ds == 'Ridge Regression':
        print('go to Random Forest Regression Algorithm')
        Ridge()
    elif ds == 'Lasso Regression':
        print('go to Neural Network Regression Algorithm')
        Lasso()
    
        
    
def browseData():
            #self.load = askopenfilename(filetypes=[("Image File",'.jpeg .jpg .png .HEIC')])
            file_path = filedialog.askopenfilename()
            filename = os.path.basename(file_path)
            t2.delete(0, tk.END)  # Clear the entry field
            t2.insert(0, filename)  # Insert the selected file path        
                
            
            
                 


message2 = tk.Label(window, text="Choose the Algorithm" ,bg="yellow"  ,fg="red"  ,width=25  ,height=1,font=('times', 15, ' bold ')) 
message2.place(x=100, y=250)


OPTIONS = sorted(list1)

S1En = tk.OptionMenu(window, dataselect,*OPTIONS)
S1En.place(x=450, y=250)


trainImg = tk.Button(window, text="Predict", command=predict  ,fg="red"  ,bg="yellow"  ,width=5  ,height=1, activebackground = "Red" ,font=('times', 15, ' bold '))
trainImg.place(x=700, y=350)


browseButton = tk.Button(window, text="Upload Test Data", command=browseData  ,fg="red"  ,bg="yellow"  ,width=25  ,height=1, activebackground = "Red" ,font=('times', 15, ' bold '))
browseButton.place(x=100, y=350)

#browseButton = tk.Button(window, text="Upload Test Data", command=browseData)
#browseButton.place(x=700, y=350)

dataentry = StringVar()
t2 = tk.Entry(width=30, bd=5, textvariable = dataentry)
t2.place(x=450, y=350)

message2 = tk.Label(window, text="Predicted Commodity Demand" ,bg="white"  ,fg="black"  ,width=30  ,height=1,font=('times', 10, ' bold ')) 
message2.place(x=100, y=450)

#out = tk.Entry(window, bd=5, textvariable=outputvar,width=25)
#out.place(x=500, y=450)


font1 = ('times', 12, 'bold')
text=tk.Text(window,height=10,width=80)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=500,y=450)
text.config(font=font1)





 
window.mainloop()