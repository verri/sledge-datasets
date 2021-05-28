import pandas as pd
from preprocess import prepare_dataset

names = ['Sex', 'Length', 'Diameter', 'Height', 'WholeWeight',
         'ShuckedWeight', 'VisceraWeight', 'ShellWeight', 'Rings']

data = pd.read_csv(
    'https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data',
    header=None,
    names=names)

data = data.drop(columns = ['Rings'])

prepare_dataset(data).to_csv('data/abalone.csv', index=False)
