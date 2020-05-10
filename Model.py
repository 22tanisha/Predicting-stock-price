import csv
import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt

dates = []
prices = []
d=28

def get_data(filename):
    
    with open(filename,'r') as csvfile:
        csvFileReader = csv.reader(csvfile)
        next(csvFileReader)      #to skip first row 
        for row in csvFileReader:
            dates.append(int(((row[0]).split("-"))[0])) 
            prices.append(float(row[1]))
        print (dates,prices)
        return

def predict_prices(dates,prices,x):
    dates = np.reshape(dates,(len(dates),1))
    prices = np.reshape(prices,(len(prices),1))

    svr_lin = SVR(kernel='linear', C=1e3)
    svr_poly = SVR(kernel='poly', C=1e3, degree=2)
    svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
    svr_lin.fit(dates, prices)    #Generates constants and co-efficients
    svr_poly.fit(dates, prices)    
    svr_rbf.fit(dates, prices)

    plt.scatter(dates, prices ,color='black', label='Data')
    plt.plot(dates, svr_rbf.predict(dates), color='red', label='RBF model')
    plt.plot(dates, svr_lin.predict(dates), color='green', label='Linear model')
    plt.plot(dates, svr_poly.predict(dates), color='blue', label='Polynomial model')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Support Vector Regression')
    plt.legend()
    #plt.show()

    return svr_rbf.predict(x)[0], svr_lin.predict(x)[0], svr_poly.predict(x)[0]

get_data('apple.csv')
predicted_price = predict_prices(dates, prices, 28)
plt.plot(d, predicted_price[0], 'ro', color='red', label='predicted')
plt.plot(d, predicted_price[1], 'ro', color='green', label='predicted')
plt.plot(d, predicted_price[2], 'ro', color='blue', label='predicted')
plt.show()
print('\nThe stock open price for 28th july is:')
print('RBF kernel:',int(predicted_price[0]))
print('Linear kernel:', int(predicted_price[1]))
print('Polynomial kernel:', int(predicted_price[2]))
