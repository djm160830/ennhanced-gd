import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import time
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif

np.random.seed(0)

"""
This class for creates a model for linear regression using the adam optimizer. 
Methods: fit, predict, gradient_descent_adam
"""
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
		tme = time.perf_counter()
		i=0
		time_cutoff=300.0

		while time.perf_counter()-tme<=time_cutoff:
			i+=1
			self.hypothesis = np.dot(x, self.theta)
			error = self.hypothesis - y
			
			mse = np.square(error).mean()
			# print(f'Iteration {i}	|	MSE: {mse}	|	R^2: {r2_score(y, self.hypothesis)}	|	time: {time.perf_counter()-tme}')
			partial = np.dot(x.T, error)/t
			m = (1-beta_1)*partial + beta_1*m
			v = (1-beta_2)*(partial**2) + beta_2*v
			m_hat = m/(1-beta_1)
			v_hat = v/(1-beta_2)
			self.theta = self.theta - (self.step_size*m_hat)/(np.sqrt(v_hat)+epsilon)

		return self.hypothesis

	def predict(self, X):
		return np.dot(X, self.theta)



'''
This method preprocesses the dataset by removing null/NA values, redundant rows, non-predictive columns, and adds
4 date-related columns.
'''
def preprocess(d):
	# Turn into dataframe
	df = pd.read_csv(d)

	# Fixing the names of some columns that have a space in the front
	df.columns = [colname.strip(' ') for colname in df.columns]

	# Remove null or NA values. The particular dataset I'm using does not have null/NA values.
	if df.isnull().values.any(): df.dropna()

	# Remove redundant rows. The particular dataset I'm using does not have redundant rows.
	if df.duplicated().sum(): df.drop_duplicates()

	# Remove noise/outlier instances
	df = df[df['n_non_stop_words']!=1042]
	df = df[df['n_tokens_content']!=0]
	Q1_shares = df['shares'].quantile(0.25)
	Q3_shares = df['shares'].quantile(0.75)
	IQR = Q3_shares-Q1_shares
	cutoff = Q3_shares + (1.5*IQR)
	df.drop(df[df['shares']>cutoff].index, inplace=True)

	# Remove non-predictive/not useful columns
	df.drop(columns=['url', 'timedelta', 'n_non_stop_words'], axis=1, inplace=True)

	return df

'''
This method retrieves the feature variables and target variable, performs feature selection to select the top 20 most
significant features, and splits the dataset into an 80/20 ratio.
'''
def split(d):
	# Get X components
	X = d.loc[:, d.columns!='shares']

	# Get target variable
	Y = d.loc[:, d.columns=='shares']

	# Feature selection using univariate selection
	selector = SelectKBest(score_func=f_classif, k=10)
	fit = selector.fit(X,Y.values.ravel())
	scores = pd.DataFrame(fit.scores_)
	cols = pd.DataFrame(X.columns)
	best_features = pd.concat([cols,scores],axis=1)
	best_features.columns=['Feature', 'Score']
	best_features = best_features.sort_values(by='Score', ascending=False).head(20)
	X = X[best_features['Feature']]

	return train_test_split(X, Y, test_size=0.2, random_state=0)


"""
This driver code reads the dataset, preprocesses it, scales it, creates the model, and predicts the target variable y.
"""
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


	print('\n\nY PREDICTION: {y_predict}\n')
	print(f"\nError: {y_predict - Y_test} \n\nMSE: {np.square(y_predict - Y_test).mean()}\n\nR^2: {r2_score(Y_test, y_predict)}")
