# NAME:			Darla Maneja (djm160830@utdallas.edu)
# SECTION:		CS4371.001
# DATE:			Sept. 12, 2020
# ASSIGNMENT 1

import numpy as np
from RawData import *
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler

import pdb

np.random.seed(0)

class LinearRegression:
	def __init__(self, hypothesis=None, theta=np.random.rand(2), step_size=0.01):
		# self.name=name // example
		self.hypothesis=hypothesis
		self.theta=theta # Start with random values 
		self.step_size=step_size


	"""
	Calculates the hypothesis (h). Compact representation shown in class was the dot product of theta (parameters) and X components.
	"""
	def fit(self, x, y):
		self.gradient_descent()
		# self.hypothesis = self.y_intercept + self.slope*x 
	

	"""
	Calculates the optimal values for parameters (theta_0 and theta_1).
	"""
	def gradient_descent(self):
		log_params=[self.theta]
		self.theta=3
		print("self.theta now:", self.theta)
		log_mse=[]




	

if __name__=='__main__':
	df = preprocess("https://raw.githubusercontent.com/djm160830/ennhanced-gd/master/OnlineNewsPopularity.csv?token=AJHKVR27TLSJYZRV5PV6AHS7MZBOU")

	X_train, X_test, Y_train, Y_test = split(df)
	# print(df.shares.describe())
	# print(X_train.info())
	# print(X_train.describe())

	# Scale
	scaler = RobustScaler()
	X_train_scaled = X_train.copy()
	X_test_scaled = X_test.copy()

	# col_names = [col for col in X_train_scaled.columns if col not in ['date', 'year', 'month', 'day']]
	col_names = X_train_scaled.columns
	train_features = X_train_scaled[col_names]
	test_features = X_test_scaled[col_names]
	scaler.fit(train_features.values)

	train_features = scaler.transform(train_features.values)
	X_train_scaled[col_names] = train_features

	test_features = scaler.transform(test_features.values)
	X_test_scaled[col_names] = test_features


	model = LinearRegression()
	model.fit(X_train_scaled, Y_train)

	pdb.set_trace()




		
		

