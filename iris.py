import pandas as pd
from preprocess import prepare_dataset

names = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']

data = pd.read_csv(
    'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',
    header=None,
    names=names)

data = data.drop(columns=['Species'])

prepare_dataset(data, nclasses=3).to_csv('data/iris.csv', index=False)
