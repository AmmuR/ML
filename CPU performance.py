# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 14:07:32 2019

@author: SUJITH KUMAR
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score,mean_squared_error

df=pd.read_csv('CPU_performance.csv')

y=df[['ERP']]

x=df.drop(['ERP'],axis=1).copy()

from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()
le.fit(df[['vender']])


df['vender']=df[['vender']].apply(lambda x:le.transform(x))
le.fit(df[['model']])

df['model']=df[['model']].apply(lambda x:le.transform(x))

####Scaling

from sklearn.preprocessing import StandardScaler

ss=StandardScaler()
ss.fit(df[['MMIN']])

df_scaled = pd.DataFrame(ss.fit_transform(df),columns = df.columns)

y=df_scaled[['ERP']]
x=df_scaled.drop(['ERP'],axis=1).copy()

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=48)
##Multilinear Regression

Mlinear=LinearRegression()
Mlinear.fit(x_train,y_train)

y_pred=Mlinear.predict(x_train)

print ("RMSE = {}".format(mean_squared_error(y_train,y_pred)))
print('R2 ={}'.format(r2_score(y_train,y_pred)))

y_predict=Mlinear.predict(x_test)
print ("RMSE = {}".format(mean_squared_error(y_test,y_predict)))
print('R2 ={}'.format(r2_score(y_test,y_predict)))

##Polynomial Regression

for degree in range(2,10):
    Polyregression=PolynomialFeatures(degree=degree)
    x_poly_train=Polyregression.fit_transform(x_train)
    
    
    poly_linear=LinearRegression()
    poly_linear.fit(x_poly_train,y_train)
    
    y_pred=poly_linear.predict(x_poly_train)

    print ("RMSE = {}".format(mean_squared_error(y_train,y_pred)))
    print('R2 ={}'.format(r2_score(y_train,y_pred)))
    
    x_poly_test=Polyregression.fit_transform(x_test)
    
    y_predict=poly_linear.predict(x_poly_test)

    print ("Test Degree = {} , RMSE {}".format(degree,mean_squared_error(y_test,y_predict)))
    print('Test R2 ={}'.format(r2_score(y_test,y_predict)))
 
##Decision Tree Regreesor

parameters={'max_depth': range(2,15),'min_samples_leaf': [1,15] }

DecTree=DecisionTreeRegressor()
GridTree=GridSearchCV(DecTree,parameters,cv=5)

GridTree.fit(x_train,y_train)
final_dt=GridTree.best_estimator_
print(final_dt)

y_pred=final_dt.predict(x_train)

print ("RMSE = {}".format(mean_squared_error(y_train,y_pred)))
print('R2 ={}'.format(r2_score(y_train,y_pred)))

y_predict=final_dt.predict(x_test)

print ("Test RMSE = {}".format(mean_squared_error(y_test,y_predict)))
print('Test R2 ={}'.format(r2_score(y_test,y_predict)))

####Random Forest Regressor

parameters={'n_estimators':[50,100],'max_depth' : [10,14], 'min_samples_split': [2,10]}

RanForest=RandomForestRegressor()
GridForest=GridSearchCV(RanForest,parameters,cv=5)

GridForest.fit(x_train,y_train)

Final_rf=GridForest.best_estimator_

print(Final_rf)
y_pred=Final_rf.predict(x_train)

print ("RMSE = {}".format(mean_squared_error(y_train,y_pred)))
print('R2 ={}'.format(r2_score(y_train,y_pred)))

y_predict=Final_rf.predict(x_test)

print ("Test RMSE = {}".format(mean_squared_error(y_test,y_predict)))
print('Test R2 ={}'.format(r2_score(y_test,y_predict)))


##Support Vector Regression###



parameters={'C':[1,25],'gamma' : ['auto',0.1],'kernel': ('linear','rbf', 'poly'),'epsilon' :[0.1],'degree':[1,4]}

svr=SVR()
GridSVR=GridSearchCV(svr,parameters,cv=5)
GridSVR.fit(x_train,y_train)

Final_svr=GridSVR.best_estimator_

print(Final_svr)
y_pred=Final_svr.predict(x_train)

print ("RMSE = {}".format(mean_squared_error(y_train,y_pred)))
print('R2 ={}'.format(r2_score(y_train,y_pred)))

y_predict=Final_svr.predict(x_test)

print ("Test RMSE = {}".format(mean_squared_error(y_test,y_predict)))
print('Test R2 ={}'.format(r2_score(y_test,y_predict)))








