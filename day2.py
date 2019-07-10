import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

data=pd.read_csv('headbrain.csv')
x=data.iloc[:,2].values
y=data.iloc[:,3].values

xMean=np.mean(x)
yMean=np.mean(y)

upper=0
lower=0
for i in range(0,len(x)):
    upper=upper+((x[i]-xMean)*(y[i]-yMean))
    lower=lower+((x[i]-xMean)**2)
bDash=upper/lower
print(bDash)

bNot=yMean-(bDash*xMean)
print(bNot)





x1=data.iloc[:,2:3].values
#from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x1,y) #this is used to train the machine 
m=regressor.coef_ 
c=regressor.intercept_
print(m)
print(c)
#print(regressor.score(x1,y))