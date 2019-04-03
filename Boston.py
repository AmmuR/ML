# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 15:38:09 2019

@author: SUJITH KUMAR
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV


from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import r2_score,mean_squared_error

from sklearn.preprocessing import StandardScaler

Boston_df=pd.read_csv('Boston.csv')

x=Boston_df[['TOWN','CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO']]
y=Boston_df[['MEDV']]

####Label Encoder-->Non-numeric to numeric###

le=LabelEncoder()
le.fit(Boston_df[['TOWN']])

print(le.classes_)

Boston_df['TOWN']=Boston_df[['TOWN']].apply(lambda x: le.transform(x))

####Scaler -->Standardize features###

ss=StandardScaler()
ss.fit(Boston_df[['TAX']])

boston_df_scaled = pd.DataFrame(ss.fit_transform(Boston_df),columns = Boston_df.columns)




#Error-->Boston_df['TAX']=Boston_df[['TAX']].apply(lambda x: le.transform(x))


x=boston_df_scaled[['TOWN','CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO']]
y=boston_df_scaled[['MEDV']]

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=40)

parameters={}

print('****Linear Regression*****')
LinearRegressor=LinearRegression()
GridLinear=GridSearchCV(LinearRegressor, parameters, cv=5)
GridLinear.fit(x_train,y_train)

y_pred=GridLinear.predict(x_train)

print("**Train**")

print ("RMSE = {}".format(mean_squared_error(y_train,y_pred)))
print('R2 ={}'.format(r2_score(y_train,y_pred)))

print("**Test**")

y_predict=GridLinear.predict(x_test)

print ("RMSE = {}".format(mean_squared_error(y_test,y_predict)))
print('R2 ={}'.format(r2_score(y_test,y_predict)))

##Polynomial Regressor##

##parameters= {'polynomialfeatures__degree':np.arange(2,10)}

##Error lines##
##poly_feat=PolynomialFeatures()
##x_poly_train = poly_feat.fit_transform(x_train)

##Polynomial_model=inearRegression()
    
##GridPoly=GridSearchCV(poly_feat, parameters, cv=5) 

##GridPoly.fit(x_poly_train,y_train)

##y_pred=GridPoly.predict(x_poly_train)

for degree in range(2,10):
    poly_feat=PolynomialFeatures(degree=degree)
    x_poly_train=poly_feat.fit_transform(x_train)
    x_poly_test=poly_feat.fit_transform(x_test)
    
    poly_regression=LinearRegression()
    poly_regression.fit(x_poly_train,y_train)
     
    y_predict=poly_regression.predict(x_poly_train)
    
    print('degree {} RMSE {}'.format(degree,mean_squared_error(y_train,y_predict)))
    print('degree {} R2 {}'.format(degree,r2_score(y_train,y_predict)))
    

    x_poly_test=poly_feat.fit_transform(x_test)
    y_pred=poly_regression.predict(x_poly_test)
    
    print('Test--degree {} RMSE {}'.format(degree,mean_squared_error(y_test,y_pred)))
    print('Test-- degree {} R2 {}'.format(degree,r2_score(y_test,y_pred)))

##DecisionTreeModel##

parameters={'max_depth': range(2,15),'min_samples_leaf': [1,15] }

DecTree=DecisionTreeRegressor()
GridDec=GridSearchCV(DecTree, parameters, cv=5)
GridDec.fit(x_train,y_train)

Final_dec_tree=GridDec.best_estimator_
y_predict=Final_dec_tree.predict(x_train)

print (Final_dec_tree)

print(' RSE {}'.format(mean_squared_error(y_train,y_predict)))
print('R2 {}'.format(r2_score(y_train,y_predict)))
 
y_pred=Final_dec_tree.predict(x_test)
    
print('Test--RMSE {}'.format(mean_squared_error(y_test,y_pred)))
print('Test-- R2 {}'.format(r2_score(y_test,y_pred)))

##Random Forest Regressor##
parameters={'n_estimators':[125,150],'max_depth' : [15,20], 'min_samples_split': [2,10]}

RandomForest=RandomForestRegressor()
GridRandom=GridSearchCV(RandomForest, parameters, cv=5)
GridRandom.fit(x_train,y_train)

Final_Random_Forest=GridRandom.best_estimator_
y_predict=Final_Random_Forest.predict(x_train)

print (Final_Random_Forest)

print(' RSE {}'.format(mean_squared_error(y_train,y_predict)))
print('R2 {}'.format(r2_score(y_train,y_predict)))
 
y_pred=Final_dec_tree.predict(x_test)
    
print('Test--RMSE {}'.format(mean_squared_error(y_test,y_pred)))
print('Test-- R2 {}'.format(r2_score(y_test,y_pred)))

##Support Vector Regressor##

from sklearn.svm import SVR

parameters={'C':[1,25],'gamma' : ['auto',0.1],'kernel': ('linear','rbf', 'poly'),'epsilon' :[0.1],'degree':[2,4]}

svr=SVR()
Gridsvr=GridSearchCV(svr,parameters,cv=5)
Gridsvr.fit(x_train,y_train)

Final_svr=Gridsvr.best_estimator_
y_pred=Final_svr.predict(x_train)

print(Final_svr)

y_pred=Final_svr.predict(x_train)

print(' RMSE {}'.format(mean_squared_error(y_train,y_pred)))
print('R2 {}'.format(r2_score(y_train,y_pred)))

y_predict=Final_svr.predict(x_test)

print(' RMSE {}'.format(mean_squared_error(y_test,y_predict)))
print('R2 {}'.format(r2_score(y_test,y_predict)))









