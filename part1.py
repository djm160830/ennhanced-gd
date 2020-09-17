# NAME:			Darla Maneja (djm160830@utdallas.edu)
# SECTION:		CS4371.001
# DATE:			Sept. 7, 2020
# ASSIGNMENT 1

import numpy as np
from PreprocessDJM import *
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

import pdb

np.random.seed(0)

class LinearRegression:
	def __init__(self, hypothesis=None, theta=None, step_size=None):
		self.hypothesis=hypothesis
		self.theta=theta 
		self.step_size=step_size


	"""
	Calculates the hypothesis (h). Compact representation shown in class was the dot product of theta (parameters) and X components.
	"""
	def fit(self, x, y):
		self.theta = np.random.rand(x.shape[1]).reshape(-1, 1) 
		x = np.asarray(x)
		y = np.asarray(y)

		self.hypothesis = self.gradient_descent_adam(x, y)
	

	"""
	Calculates the optimal values for parameters (theta_0 and theta_1).
	"""
	def gradient_descent_adam(self, x, y):
		# Recommended default values
		self.step_size=0.01
		epsilon=10**(-8)
		beta_1=0.9
		beta_2=0.999

		# Moment vectors
		m=0
		v=0

		# Time
		t=x.shape[0]

		# Logs
		log_params=[self.theta]
		log_hypothesis=[]
		log_mse=[]
		
		i=0
		# for i in range(t): 
		while True:
			i+=1
			log_hypothesis.append(np.dot(x, self.theta))
			error = log_hypothesis[-1] - y
			# log_mse.append((1/(2*t))*np.sum(error**2)) 
			log_mse.append(np.square(error).mean())
			print(f'Iteration {i}	| Cost: {log_mse[-1]}	')
			partial = np.dot(x.T, error)/t
			m = (1-beta_1)*partial + beta_1*m
			v = (1-beta_2)*(partial**2) + beta_2*v
			m_hat = m/(1-beta_1)
			v_hat = v/(1-beta_2)
			self.theta = self.theta - (self.step_size*m_hat)/(np.sqrt(v_hat)+epsilon)
			if log_mse[-1] < 100 and log_mse[-1] > -100: break

		return log_hypothesis[-1]

	def predict(self, X):
		return np.dot(X, self.theta)



if __name__=='__main__':
	df = preprocess("https://raw.githubusercontent.com/djm160830/ennhanced-gd/master/OnlineNewsPopularity.csv?token=AJHKVR27TLSJYZRV5PV6AHS7MZBOU")

	X_train, X_test, Y_train, Y_test = split(df)

	# Scale
	scaler = MinMaxScaler()
	X_train_scaled = X_train.copy()
	X_test_scaled = X_test.copy()

	col_names = X_train_scaled.columns
	train_features = X_train_scaled[col_names]
	test_features = X_test_scaled[col_names]
	scaler.fit(train_features.values)

	train_features = scaler.transform(train_features.values)
	X_train_scaled[col_names] = train_features

	test_features = scaler.transform(test_features.values)
	X_test_scaled[col_names] = test_features


	# Construct model
	model = LinearRegression()
	model.fit(X_train_scaled, Y_train)

	# Predict y
	y_predict = model.predict(X_test_scaled)

	print(f"Error: {y_predict - Y_test} | MSE: {np.square(y_predict - Y_test).mean()}")
	pdb.set_trace()




		
		

