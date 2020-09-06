import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.ar_model import AR
from statsmodels.graphics.tsaplots import plot_acf


def RMSE(y_true, y_pred):
    error = np.sqrt(mean_squared_error(y_true, y_pred))
    return error

def load_dataset(path):
    return pd.read_csv(path)

def main():
    path="./outliers_removed.csv"
    DATA = load_dataset(path)
    InBandwidth_data = DATA["InBandwidth"].values 
    
    data_t1=InBandwidth_data[1:]
    data_t=InBandwidth_data[:-1]
    plt.scatter(data_t,data_t1)
    plt.xlabel("InBandwidth(t)")
    plt.ylabel("InBandwidth(t+1)")
    plt.show()
    
    plot_acf(InBandwidth_data, lags=[i for i in range(1,20)])
    plt.xlabel("Lag")
    plt.ylabel("Correaltion")
    plt.show()
    plt.savefig("plotacf.jpg")
    train , test = train_test_split(InBandwidth_data, test_size = 0.3, shuffle=False, random_state=42)
    AutoRegression(train,test)
    

def AutoRegression(train,test):
    model = AR(train)
    model_fit = model.fit()
    print('Lag:', model_fit.k_ar)
    print('Coefficients: %s' % model_fit.params)
    predictions = model_fit.predict(start=len(train), end=len(train)+len(test)-1, dynamic=False)
    
    error = RMSE(test, predictions)
    plt.plot(test,'lightblue')
    plt.ylabel("InBandwidth")
    plt.plot(predictions,'r')
    plt.legend(["Original","Predicted"])
    plt.savefig("AutoRegressionGraph.jpg")
    plt.show()
    print("RMSE : ",error)

if __name__=='__main__':
    main()