import pandas as pd
import numpy as np
import os

def load_data(data_path, n_plants, resample_rule=None):
    """
    data_path: directory where the data is saved.
    n_plants: number of plants to load.
    resample: resample rule for data aggregation.
    """
    data = [None] * n_plants
    for path, _, file_names in os.walk(data_path):
        for i, file_name in enumerate(file_names):
            if i + 1 > n_plants:
                break
            data[i] = pd.read_csv(os.path.join(data_path, 'plant_{}'.format(i)),\
                                  index_col=0, names=['85m_speed'], parse_dates=True)
            if resample_rule:
                data[i] = data[i].resample(resample_rule).mean()

            data[i] = data[i]['85m_speed'].values

    data = np.stack(data, axis=0)
    return data
