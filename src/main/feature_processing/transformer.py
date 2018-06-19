
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import MinMaxScaler


def get_transformer(scaler_type=None):
    # Return Respective transformer to

    if scaler_type == "minmax":
        return MinMaxScaler()
    elif scaler_type == 'normalizer':
        return Normalizer()

    #Standard Scaler returned by default.
    return StandardScaler()
