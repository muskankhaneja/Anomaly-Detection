import os
import pandas as pd
import definitions
from src import data_preprocessing, anomaly_detection, ann, autoencoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
pd.set_option('display.max_columns', None)

df = pd.read_csv(os.path.join(definitions.file_loc, definitions.filename))
print("Data loaded with shape: ", df.shape)
print(df.isFraud.value_counts())
print(df.isFraud.value_counts(normalize=True))

# preprocessing
preprocess = data_preprocessing.DataPreprocessing(df)
df_preprocessed = preprocess.preprocessing()
print("Data loaded with shape: ", df_preprocessed.shape)
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

# fitting ANN to train autoencoder---------------------------------------------------------------------
# scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(df_preprocessed)

# converting to tensor
X_train = torch.tensor(X_train, dtype=torch.float32).clone().detach()

# model training
input_dim = X_train.shape[1]
encoding_dim = 32

autoencoder_model = autoencoder.Autoencoder(input_dim, encoding_dim)

autoencoder_model = autoencoder.train_autoencoder(autoencoder_model, X_train, lr=0.001, epochs=100)
errors = autoencoder.calculate_reconstruction_error(autoencoder_model, X_train)
anomalies = errors > 0.001
len(anomalies)
