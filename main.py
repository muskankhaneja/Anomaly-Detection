import os
import xml.dom

import mlflow
import mlflow.pytorch
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from datetime import datetime

import definitions
from src import data_preprocessing, autoencoder

pd.set_option('display.max_columns', None)

print("Loading dataset...")
df = pd.read_csv(os.path.join(definitions.file_loc, definitions.filename))
print("Data loaded with shape: ", df.shape)
print(df.isFraud.value_counts())
print(df.isFraud.value_counts(normalize=True))

# preprocessing
preprocess = data_preprocessing.DataPreprocessing(df)
df_preprocessed = preprocess.preprocessing()
print("Data loaded with shape: ", df_preprocessed.shape)

# fitting ANN to train autoencoder---------------------------------------------------------------------
# scaling
print("scaling..")
scaler = StandardScaler()
df_train = scaler.fit_transform(df_preprocessed)

# converting to tensor
print("converting to tensor..")
df_train_tensor = torch.tensor(df_train, dtype=torch.float32).clone().detach()

# model training
print("training model..")
input_dim = df_train_tensor.shape[1]
encoding_dim = 32
learning_rates = [0.001, 0.01]
n_epochs_list = [20, 50]

for learning_rate in learning_rates:
    for n_epochs in n_epochs_list:
        autoencoder_model = autoencoder.Autoencoder(input_dim, encoding_dim)

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        with mlflow.start_run(run_name=f'run_{timestamp}'):
            mlflow.log_params({"num_epochs": n_epochs, "learning rate": learning_rate})
            autoencoder.train_autoencoder(autoencoder_model, df_train_tensor, lr=learning_rate, epochs=n_epochs)

        print(f"training completed for {learning_rate} and {n_epochs}")

# load experiment with the least loss
client = mlflow.tracking.MlflowClient()
runs = client.search_runs(experiment_ids=['0'])

min_loss_value = np.inf
for run in runs:
    loss_val = run.data.metrics.get('loss')
    if loss_val < min_loss_value:
        min_loss_value = loss_val
        min_loss_run_id = run.info.run_id

model_path = f"runs:/{min_loss_run_id}/models"
loaded_model = mlflow.pytorch.load_model(model_path)
print(f"Loaded model from run with least loss (Run ID: {min_loss_run_id})")

# evaluating results
errors = autoencoder.calculate_reconstruction_error(loaded_model, df_train_tensor)
df['errors'] = pd.Series(errors)

# checking distribution of errors
print(df.groupby('isFraud')['errors'].describe().round(2).T)

# checking if top 8213 obs based on errors sorted by desc are actually fraud
print(df.sort_values(by='errors', ascending=False).head(8213)['isFraud'].value_counts())

# lift table
df['bin'] = pd.qcut(df['errors'], q=100, labels=False)
df_lift = df.groupby('bin').agg({'isFraud': ['count', 'sum'], 'errors': 'mean'}).sort_values(by='bin', ascending=False).reset_index()
df_lift.columns = df_lift.columns.map(lambda col: '_'.join(col))

total_frauds = len(df[df['isFraud'] == 1])
df_lift['perc_isFraud'] = df_lift['isFraud_sum']/total_frauds
df_lift['cum_perc_isFraud'] = df_lift['perc_isFraud'].cumsum()

print("Lift table")
print(df_lift)
df_lift['cum_perc_isFraud'].plot()


# # anomaly detection using Isolation forest------------------------------------------------------------
# iso_forest = anomaly_detection.TrainIsolationForest(df_preprocessed)
# iso_model = iso_forest.iso_fit(contamination=0.001)
#
# # get anomaly scores
# df_iso = anomaly_detection.predict(iso_model, df)
#
# # checking
# print(df_iso['isFraud'].value_counts())
# print(df_iso[df_iso['anomaly_scores'] < 0].shape)
# print(df_iso[df_iso['anomaly_scores'] < 0]['isFraud'].value_counts())
# df_iso['anomaly_scores'].hist()
#
# # fitting ANN to train classifier---------------------------------------------------------------------
# X = df_preprocessed
# y = df['isFraud']
#
# # train test split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2023)
#
# # scaling
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)
#
# # converting to tensor
# X_train, y_train, X_test, y_test = ann.convert_to_tensor(X_train, y_train, X_test, y_test)
#
# # model training
# model = ann.ArtificialNeuralNetwork(X_train.shape[1], 64, 1)
# model = ann.train_ann(model, X_train, y_train, lr=0.001, epochs=100)
#
# # model evaluation
# with torch.no_grad():
#     model.eval()
#     test_outputs = model(X_test)
#     predicted_labels = (test_outputs >= 0.5).float()
#     accuracy = (predicted_labels == y_test.view(-1, 1)).float().mean()
#
# print(f'Test Accuracy: {accuracy.item():.4f}')
