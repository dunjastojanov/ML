import sys

import numpy as np
import pandas as pd
from sklearn.metrics import v_measure_score
from sklearn.mixture import GaussianMixture

more = {
    "da": 1,
    "ne": 0
}


def load_file(file_name):
    return pd.read_csv(file_name)


def drop_column_with_name(data, name):
    data = data.drop(name, axis=1)
    return data


def remove_row_with_nan_atr(data, count_of_nan_atr=2):
    mask = data.isna().sum(axis=1) > count_of_nan_atr
    return data.drop(data[mask].index)


def remove_row_for_value(data, atr_name, value):
    data = data.drop(data[(data[atr_name] > value)].index)
    return data


def remove_outlier_where_y_is_nan(data):
    return data.drop(data[(data['obrazovanje'].isna())].index)


def fill_in_izvoz(data):
    median_izvoz = data['Izvoz'].mean()
    data['Izvoz'] = data['Izvoz'].fillna(median_izvoz)
    return data

def fill_in_BDP(data):
    median_izvoz = data['BDP'].mean()
    data['BDP'] = data['BDP'].fillna(median_izvoz)
    return data

def normalize_other_atr(data, atr_name, map):
    data[atr_name] = data[atr_name].map(map)
    return data


def smarter_version_one_hot(data, atr_name):
    df_dummy = pd.get_dummies(data[atr_name], prefix=atr_name).iloc[:, :-1]
    new_df = pd.concat([data, df_dummy], axis=1)
    new_df.drop([atr_name], axis=1, inplace=True)
    return new_df


def create_normalization_map(dataframe):
    map = {}
    for column in dataframe.columns:
        if column in ['BDP', 'Inflacija', 'Izvoz']:
            map[column] = {
                'min': dataframe[column].min(),
                'max': dataframe[column].max()
            }
    return map


def normalize_dataframe(data, map_for_normalization):
    for column in map_for_normalization.keys():
        data[column] = (data[column] - map_for_normalization[column]["min"]) / (
                map_for_normalization[column]["max"] - map_for_normalization[column]["min"])
    return data


def preprocess_dataframe(data):
    data = remove_row_with_nan_atr(data, count_of_nan_atr=1)
    data = fill_in_izvoz(data)
    data = fill_in_BDP(data)
    data = normalize_other_atr(data, 'More', more)
    data = smarter_version_one_hot(data, 'Religija')
    normalization_map = create_normalization_map(data)
    data = normalize_dataframe(data, normalization_map)
    return data


def preprocess_test_dataframe(data, normalization_map):
    data = normalize_other_atr(data, 'More', more)
    data = smarter_version_one_hot(data, 'Religija')
    data = normalize_dataframe(data, normalization_map)
    return data


def gaussian(test_data, train_data):
    num_clusters = 4
    x_train = drop_column_with_name(train_data, 'Region')

    y_test = test_data['Region']
    x_test = drop_column_with_name(test_data, 'Region')

    # tol parametar
    # GridSearch

    gmm = GaussianMixture(n_components=num_clusters, covariance_type='diag', init_params='kmeans', random_state=42)
    gmm.fit(x_train)
    predicted_clusters = gmm.predict(x_test)

    print(v_measure_score(y_test, predicted_clusters))


def add_missing_columns(test, column_names):
    dataframe = pd.DataFrame(columns=column_names)
    for name in column_names:
        if name not in test.columns:
            dataframe[name] = pd.Series(np.zeros(test.shape[0]))
        else:
            dataframe[name] = test[name]
    return dataframe


if __name__ == '__main__':
    train_data = load_file(sys.argv[1])
    train_data = preprocess_dataframe(train_data)

    test_data = load_file(sys.argv[2])
    test_data = preprocess_test_dataframe(test_data, create_normalization_map(train_data))
    test_data = add_missing_columns(test_data, train_data.columns)

    gaussian(test_data, train_data)
