import pandas as pd

class RawData:
	def __init__(self, raw=None):
		self.raw=raw

	def preprocess(self):
		# Turn into dataframe
		df = pd.read_csv(self.raw)

		# Remove null or NA values. The particular dataset I'm using does not have null/NA values.
		if df.isnull().values.any(): df.dropna()

		# Remove redundant rows. The particular dataset I'm using does not have redundant rows.
		if df.duplicated().sum(): df.drop_duplicates()
	
		# Convert categorical variables to numerical variables. The particular dataset I'm using does not have categorial variables.

		# Any other pre-processing that you may need to perform
		# Extract dates from URLs
		df[['date', 'year', 'month', 'day']] = df['url'].str.extract(r'\S*((\d{4})\/(\d{2})\/(\d{2}))\S*')
		df['date'] = pd.to_datetime(df['date'], format='%Y/%m/%d')

		# If you feel an attribute is not suitable or is not correlated with the outcome, you might want to get rid of it
		# Remove non-predictive columns
		df.drop(columns=['url', ' timedelta'])

		return df

