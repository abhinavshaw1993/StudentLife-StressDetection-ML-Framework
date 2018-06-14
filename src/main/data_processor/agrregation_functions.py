# imports for aggregations functions
from scipy.stats import iqr as quartile_range
from scipy.stats import kurtosis
from scipy.fftpack import fft
import math
import pandas as pd
import numpy as np


def linear_fit(array_like):
    # Linear features
    if (len(array_like) == 0):
        return [0, 0]
    p = np.polyfit(np.arange(len(array_like)), array_like, 1)
    return [p[0], p[1]]


def poly_fit(array_like):
    # Poly features
    if (len(array_like) == 0):
        return [0, 0, 0]
    p = np.polyfit(np.arange(len(array_like)), array_like, 2)
    return [p[0], p[1], p[2]]


def iqr(array_like):
    # inter quartile range.
    result = quartile_range(array_like)
    return result if not math.isnan(result) else 0


def kurt(array_like):
    result = kurtosis(array_like)
    return result if not math.isnan(result) else 0


def mcr(array_like):
    # returns how many times the mean has been crossed.
    mean = np.mean(array_like)
    array_like = array_like - mean
    return np.sum(np.diff(np.sign(array_like)).astype(bool))


def fourier_transform(array_like):
    # Return Fast Fourier transfor of array.
    result = fft(array_like)
    return 0


def extend_complex_features(feature_name, resampled_df, columns=None):
    # This function adds liear features such as slope and intercept to the dataset.
    # If columns is None then apply features to all columns.
    function_mapper = {

        'linear': (linear_fit, ['linear_m', 'linear_c']),
        'poly': (poly_fit, ['poly_a', 'poly_b', 'poly_c']),
        'iqr': (iqr, []),
        'kurtosis': (kurt, []),
        'mcr': (mcr, []),
        'fft': (fourier_transform, [])

    }
    # please give empty list if no columns name required.

    if columns:
        complex_cols = [(feature_name, f) for f in columns]
    else:
        complex_cols = [(feature_name, f) for f in resampled_df.columns.values]

    complex_feature = resampled_df.agg({feature_name: function_mapper[feature_name][0]})

    if len(function_mapper[feature_name][1]) > 0:
        complex_feature = pd.concat(
            [pd.DataFrame(
                complex_feature[f].values.tolist(),
                columns=[(col, f[1]) for col in function_mapper[feature_name][1]],
                index=complex_feature.index) for f in complex_cols],
            axis=1
        )

    return complex_feature