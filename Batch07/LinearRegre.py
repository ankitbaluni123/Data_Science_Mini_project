import pandas as pd
import matplotlib.pyplot as plt
from numpy.polynomial.polynomial import polyfit
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,accuracy_score,r2_score
from math import sqrt
from sklearn.preprocessing import PolynomialFeatures
def rms(y_test,y_pred): return sqrt(mean_squared_error(y_test,y_pred))

file=["1","2","3","4","5","6"]
for i in file:
    data=pd.read_csv("dim_reduced_to_"+i+".csv")
    print("For dim_reduced_to_"+i+".csv ")
    data.drop("CreationTime",inplace=True,axis=1)
    col=data["InBandwidth"]
    data.drop("InBandwidth",inplace=True,axis=1)
    X_train, X_test, Y_train, Y_test = train_test_split(data,col,test_size=0.33, random_state=42)
    degrees=[2,3,4]
    for i in degrees:
        regressor=PolynomialFeatures(degree = i)
        x_poly=regressor.fit_transform(X_train)
        regressor.fit(x_poly,Y_train)
        lin_reg=LinearRegression()
        lin_reg.fit(x_poly,Y_train)
        Y_pred_test=lin_reg.predict(regressor.fit_transform(X_test))
        plt.scatter(Y_test,Y_pred_test)
        plt.xlabel("Y_test")
        plt.ylabel("Y_pred_test")
        plt.show()
        print("R2 score when degree =",i,"is: ",r2_score(Y_test.values,Y_pred_test))
        print("rmse score when degree=",i,"is:",rms(Y_test.values,Y_pred_test))
        Y_pred_train=lin_reg.predict(regressor.fit_transform(X_train))
        plt.scatter(Y_test,Y_pred_test)
        plt.xlabel("Y_train")
        plt.ylabel("Y_pred_train")
        plt.show()
        print("R2 score when degree =",i,"is: ",r2_score(Y_train.values,Y_pred_train))
        print("rmse score when degree=",i,"is:",rms(Y_train.values,Y_pred_train))
        print("---------------------------------------------------")
    print("**************************************************************************")