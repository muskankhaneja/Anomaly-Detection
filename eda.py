import pandas as pd
import definitions
import os

pd.set_option('display.max_columns', None)

# read data
df = pd.read_csv(os.path.join(definitions.file_loc, definitions.filename))
print("Data loaded with shape: ", df.shape)

# check info
print(df.info())

# target distribution
print(df['isFraud'].value_counts())
print(df['isFraud'].value_counts(normalize=True))

# The dataset comprises transactions which occurred over a simulated time span of 30 days.
# print snapshot of data
print(df.head())

# numerical
num_col = df.select_dtypes(exclude='object').columns
print("Numerical columns: ", num_col)

# summary
print(df[num_col].describe().round(2).T)

# categorical
cat_col = df.select_dtypes(include='object').columns
print("Categorical columns: ", cat_col)
print(df[cat_col].describe(include='object'))

# Feature engineering
# Checking if difference in balance is correctly recorded
df['diffbalanceOrg'] = df['newbalanceOrig'] - df['oldbalanceOrg']
print(df['diffbalanceOrg'].corr(df['amount']))

df['diffbalanceDest'] = df['newbalanceDest'] - df['oldbalanceDest']
print(df['diffbalanceDest'].corr(df['amount']))


# getting hour of the day from step
df['hour'] = df['step'] % 24

# Categorical features
print(df['type'].value_counts())


