from sklearn.ensemble import IsolationForest
from src import data_preprocessing
import seaborn as sns


class TrainIsolationForest:

    def __init__(self, df):
        self.df = df
        return

    def fit(self, contamination=0.01, random_state=2023):
        forest = IsolationForest(contamination=contamination, random_state=random_state)
        forest.fit(self.df)
        print("Training complete..")
        return forest


def predict(forest, df):
    preprocessing = data_preprocessing.DataPreprocessing(df)
    df_preprocessed = preprocessing.preprocessing()

    scores = forest.score_samples(df_preprocessed)
    print('Scoring complete..')
    df['anomaly_scores'] = scores

    # checking distribution
    print(df['anomaly_scores'].describe().round(2))
    sns.histplot(x=scores)

    # checking description of scores
    print(df.groupby('isFraud')['anomaly_scores'].describe().T.round(2))
    sns.boxplot(data=df, x='anomaly_scores', hue='isFraud')

    return df
