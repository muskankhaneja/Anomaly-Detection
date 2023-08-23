import os

import mlflow
import mlflow.pytorch
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

# df_preprocessed = df_preprocessed.head(1000)

# fitting ANN to train autoencoder---------------------------------------------------------------------
# scaling
print("scaling..")
scaler = StandardScaler()
X_train = scaler.fit_transform(df_preprocessed)

# converting to tensor
print("converting to tensor..")
X_train = torch.tensor(X_train, dtype=torch.float32).clone().detach()

print("training model..")

# model training
input_dim = X_train.shape[1]
encoding_dim = 32
learning_rates = [0.001, 0.01]
n_epochs_list = [20, 50]
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

for learning_rate in learning_rates:
    for n_epochs in n_epochs_list:
        autoencoder_model = autoencoder.Autoencoder(input_dim, encoding_dim)

    with mlflow.start_run(run_name=f'run_{timestamp}'):
        mlflow.log_params({"num_epochs": n_epochs, "learning rate": learning_rate})
        autoencoder.train_autoencoder(autoencoder_model, X_train, lr=learning_rate, epochs=n_epochs)

    print(f"training completed for {learning_rate} and {n_epochs}")


# errors = autoencoder.calculate_reconstruction_error(autoencoder_model, X_train)
# anomalies = errors > 0.001
# len(anomalies)

#
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
