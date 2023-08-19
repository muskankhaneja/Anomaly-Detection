import pandas as pd
import numpy as np
import definitions


class DataPreprocessing:

    def __init__(self, df):
        self.df = df

    def preprocessing(self):
        # exclude ground truth

        # feature engineering
        # numerical columns
        num_col = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
        features = self.df[num_col].copy()

        features['diffbalanceOrg'] = features['newbalanceOrig'] - features['oldbalanceOrg']
        features['diffbalanceDest'] = features['newbalanceDest'] - features['oldbalanceDest']

        features['hour'] = self.df['step'] % 24

        # categorical columns
        type_one_hot = pd.get_dummies(self.df['type'], dtype=int)
        features = pd.concat([features, type_one_hot], axis=1)

        return features
