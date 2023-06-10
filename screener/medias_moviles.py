import numpy as np
import pandas as pd


def get_moving_averages(data, label='close', window=55):
    dataux = pd.DataFrame()
    dataux[f"SMA_{window}"] = data[label].rolling(window).mean()

    def wma(serie):
        n = len(serie)
        f = [i / sum(range(n + 1)) for i in range(1, n + 1)]
        return np.array(serie).dot(f)

    dataux[f"EMA_{window}"] = data[label].ewm(span=window).mean()
    dataux[f"WMA_{window}"] = data[label].rolling(window).apply(wma)
    dataux[f"DEMA_{window}"] = (
        dataux[f"EMA_{window}"] * 2 - dataux[f"EMA_{window}"].ewm(span=window).mean()
    )
    dataux[f"TRIMA_L_{window}"] = dataux[f"SMA_{window}"].rolling(window).mean()
    dataux[f"TRIMA_E_{window}"] = dataux[f"EMA_{window}"].ewm(span=window).mean()
    dataux[f"TEMA_{window}"] = (
        dataux[f"EMA_{window}"] * 3
        - dataux[f"TRIMA_E_{window}"] * 3
        + dataux[f"TRIMA_E_{window}"].ewm(span=window).mean()
    )

    dataux.drop(columns=f"TRIMA_E_{window}", inplace=True)

    return dataux
