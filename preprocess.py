import pandas as pd
import numpy as np
import math

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, KBinsDiscretizer


def prepare_dataset(data, nclasses=None):

    nbins = nclasses if nclasses is not None else math.ceil(
        math.log10(data.shape[0]))

    # TODO: binary features must not be encoded
    categorical = data.dtypes == 'object'
    numeric = data.dtypes != 'object'

    discretizer = KBinsDiscretizer(n_bins=nbins, encode='onehot-dense',
                                   strategy='quantile')
    onehot = OneHotEncoder(sparse=False)

    transformer = ColumnTransformer([
        ('discretizer', discretizer, numeric),
        ('onehot', onehot, categorical)]) # TODO: verify passthrough

    array = transformer.fit_transform(data)

    if np.sum(numeric) > 0:
        edges = transformer.transformers_[0][1].bin_edges_
        numeric_names = data.columns[numeric]
        numeric_names = [
            f'{numeric_names[j]} in ({edges[j][b]}, {edges[j][b+1]}]' for j in range(
                len(numeric_names)) for b in range(nbins)]
    else:
        numeric_names = []

    if np.sum(categorical) > 0:
        categorical_names = transformer.transformers_[
            1][1].get_feature_names(data.columns[categorical])
    else:
        categorical_names = []

    return pd.DataFrame(array, columns=np.concatenate(
        (numeric_names, categorical_names)), dtype=int)
