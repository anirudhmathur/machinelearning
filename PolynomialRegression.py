import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

F, N = map(int,input().split())
trainingData = np.array([input().split() for _ in range(N)], float)
T = int(input())
testData = np.array([input().split() for _ in range(T)], float)

X = trainingData[:,0:-1]
y = trainingData[:,-1]

poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
linear_reg = LinearRegression()
linear_reg.fit(X_poly,y)

testData_predict = linear_reg.predict(poly_reg.fit_transform(testData))

print(*testData_predict,sep='\n')
