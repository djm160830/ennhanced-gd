import pandas as pd
from sklearn.model_selection import train_test_split

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

	# Convert categorical variables to numerical variables. The particular dataset I'm using does not have categorial variables.

	# Any other pre-processing that you may need to perform
	# Extract dates from URLs
	# df[['date', 'year', 'month', 'day']] = df['url'].str.extract(r'\S*((\d{4})\/(\d{2})\/(\d{2}))\S*')
	# df['date'] = pd.to_datetime(df['date'], format='%Y/%m/%d')

	# If you feel an attribute is not suitable or is not correlated with the outcome, you might want to get rid of it
	# Remove non-predictive columns
	df.drop(columns=['url', 'timedelta'], inplace=True)

	return df

'''
This method retrieves the feature variables and target variable, and splits the dataset into an 80/20 ratio.
'''
def split(d):
	# Get X components
	X = d.loc[:, d.columns!='shares']

	# Get target variable
	Y = d.loc[:, d.columns=='shares']

	return train_test_split(X, Y, test_size=0.2, random_state=0)

