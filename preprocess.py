import pandas as pd
import numpy as np
import math

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, KBinsDiscretizer

def prepare_dataset(data, nclasses=None):

    nbins = nclasses if nclasses is not None else math.ceil(math.log10(data.shape[0]))

    # TODO: binary features must not be encoded
    categorical = data.dtypes == 'object'
    numeric = data.dtypes != 'object'

    discretizer = KBinsDiscretizer(n_bins=nbins, encode='onehot-dense',
            strategy='quantile')
    onehot = OneHotEncoder(sparse=False)

    transformer = ColumnTransformer([
            ('discretizer', discretizer, numeric),
            ('onehot', onehot, categorical)])

    array = transformer.fit_transform(data)

    edges = transformer.transformers_[0][1].bin_edges_
    numeric_names = data.columns[numeric]
    # TODO: check if [a, b) is correct (sklearn does not document it very well)
    numeric_names = [ f'{numeric_names[j]} in [{edges[j][b]}, {edges[j][b+1]})' for b in range(nbins) for j in range(len(numeric_names))]

    categorical_names = transformer.transformers_[1][1].get_feature_names(data.columns[categorical])

    return pd.DataFrame(array, columns=np.concatenate((numeric_names,
        categorical_names)), dtype=int)
