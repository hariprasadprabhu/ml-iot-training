import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

#data=pd.read_csv('headbrain.csv')
#x=data["Head Size(cm^3)"].values#use by name 
#y=data.iloc[:,3].values #use by index use

data=pd.read_csv('Salary_Data.csv')
x=data.iloc[:,0:1].values #makes it 2 dimentionsl 
y=data.iloc[:,1].values
#plt.plot(x,y)
#plt.scatter(x,y)
#using ordinary least square method is used
#%matplotlib auto


from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=.2,random_state=0)


from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train) #this is used to train the machine 
m=regressor.coef_ 
c=regressor.intercept_
#kk=input("enter the value") 
#l=kk.split()
#print("year\tsalary")
#for i in l:
#    res=regressor.predict([[float(i)]]) #to find the salary for given input 
#    print(float(i),"\t",res)
plt.scatter(x_train,y_train,color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.scatter(x_train,regressor.predict(x_train),color='green')
res=0
y_pred=regressor.predict(x_test)
for i in range(0,len(y_test)):
    res=res+(y_test[i]-y_pred[i])**2
ans=res/len(y_test)
finalres=math.sqrt(ans)
print(finalres)


from sklearn.metrics import mean_squared_error
mse=mean_squared_error(y_test,y_pred)**(1/2)
score=regressor.score(x_test,y_test)


