import os
import pandas as pd
import seaborn as sns

import definitions, data_preprocessing

from sklearn.ensemble import IsolationForest

pd.set_option('display.max_columns', None)

df = pd.read_csv(os.path.join(definitions.file_loc, definitions.filename))
print("Data loaded with shape: ", df.shape)
print(df.isFraud.value_counts())
print(df.isFraud.value_counts(normalize=True))

# preprocessing
preprocess = data_preprocessing.DataPreprocessing(df)
df_preprocessed = preprocess.preprocessing()
print("Data loaded with shape: ", df_preprocessed.head())

# Fit model
forest = IsolationForest(contamination=0.01, random_state=2023)
forest.fit(df_preprocessed)
print("Executed")

# Predictions
scores = forest.score_samples(df_preprocessed)
df['anomaly_scores'] = scores
print(df['anomaly_scores'].describe().round(2))

# visualizing predictions
sns.histplot(x=scores)

# checking description of scores
sns.boxplot(data=df, x='anomaly_scores', hue='isFraud')
print(df.groupby('isFraud')['anomaly_scores'].describe().T.round(2))

