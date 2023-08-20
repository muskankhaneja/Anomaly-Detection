import os
import pandas as pd
import definitions
from src import data_preprocessing, anomaly_detection

pd.set_option('display.max_columns', None)

df = pd.read_csv(os.path.join(definitions.file_loc, definitions.filename))
print("Data loaded with shape: ", df.shape)
print(df.isFraud.value_counts())
print(df.isFraud.value_counts(normalize=True))

# preprocessing
preprocess = data_preprocessing.DataPreprocessing(df)
df_preprocessed = preprocess.preprocessing()
print("Data loaded with shape: ", df_preprocessed.head())

# anomaly detection using Isolation forest
iso_forest = anomaly_detection.TrainIsolationForest(df_preprocessed)
iso_model = iso_forest.fit()

# get anomaly scores
df_iso = anomaly_detection.predict(iso_model, df)




# fitting ANN to train classifier
X = df_preprocessed
y = df['isFraud']

# split data into train test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2023)

# Scale
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# converting to tensor
import torch
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_train, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)
