import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

data=pd.read_csv('headbrain.csv')
x=data.iloc[:,2].values
y=data.iloc[:,3].values

xMean=np.mean(x)
yMean=np.mean(y)
x1=data.iloc[:,2:3].values
#from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x1,y,test_size=.2,random_state=0)
y_testMean=np.mean(y_test)
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train) #this is used to train the machine 
y_pred=regressor.predict(x_test)
SSt=0
SSr=0
for i in range(0,len(y_test)):
    SSt=SSt+((y_test[i]-y_testMean)**2)
    SSr=SSr+(y_test[i]-y_pred[i])**2
r2=1-(SSr/SSt)
print(r2)
print(regressor.score(x_test,y_test))