import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

minium_years = {
    "osnovne studije": 23.0,
    "master": 28.0,
    "doktorat": 31.0,
    "srednja skola": 18.0
}

zdravstveno_osiguranje = {
    "da": 1,
    "ne": 0
}

tip_posla = {
    "informacioni": 1,
    "industrijski": 0
}


def load_file(file_name):
    return pd.read_csv(file_name)


def remove_row_with_nan_atr(data, count_of_nan_atr=2):
    mask = data.isna().sum(axis=1) > count_of_nan_atr
    return data.drop(data[mask].index)


def remove_outlier_by_age(data):
    for key in minium_years.keys():
        data = data.drop(data[(data['godine'] < minium_years.get(key)) & (
                data['obrazovanje'] == key)].index)
    return data


def remove_outlier_where_y_is_nan(data):
    return data.drop(data[(data['obrazovanje'].isna())].index)


def create_normalization_map(dataframe):
    map = {}
    for column in dataframe.columns:
        if column in ['plata', 'godine']:
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


def smarter_version_one_hot(data, atr_name):
    df_dummy = pd.get_dummies(data[atr_name], prefix=atr_name).iloc[:, :-1]
    new_df = pd.concat([data, df_dummy], axis=1)
    new_df.drop([atr_name], axis=1, inplace=True)
    return new_df


def fill_in_age(data):
    mean_age = data['godine'].mean()
    data['godine'] = data['godine'].fillna(mean_age)
    return data


def fill_with_most_frequent(data, atr_name):
    most_frequent_atr = data[atr_name].mode()[0]
    data[atr_name].fillna(most_frequent_atr, inplace=True)


def fill_in_salary(data):
    mean_age = data['plata'].median()
    data['plata'] = data['plata'].fillna(mean_age)
    return data


def normalize_other_atr(data, atr_name, map):
    data[atr_name] = data[atr_name].map(map)
    return data


def drop_column_with_name(data, name):
    data = data.drop(name, axis=1)
    return data


def add_rows(dataframe, percentage=0.1):
    for key in minium_years.keys():
        filtered_df = dataframe[dataframe['obrazovanje'] == key]
        new_row = {
            'godine': filtered_df['godine'].median(),
            'bracni_status': filtered_df['bracni_status'].mode()[0],
            'rasa': filtered_df['rasa'].mode()[0],
            'tip_posla': filtered_df['tip_posla'].mode()[0],
            'zdravstveno_osiguranje': filtered_df['zdravstveno_osiguranje'].mode()[0],
            'plata': filtered_df['plata'].median(),
            'obrazovanje': key
        }
        stop = int(len(filtered_df) * percentage)
        for i in range(stop):
            dataframe.append(new_row, ignore_index=True)
    return dataframe


def preprocess_dataframe(data):
    data = remove_row_with_nan_atr(data)
    data = remove_outlier_where_y_is_nan(data)
    data = fill_in_age(data)
    data = remove_outlier_by_age(data)
    data = fill_in_salary(data)
    data = drop_column_with_name(data, 'bracni_status')
    data = drop_column_with_name(data, 'zdravstveno_osiguranje')
    normalization_map = create_normalization_map(data)
    data = normalize_dataframe(data, normalization_map)
    fill_with_most_frequent(data, 'rasa')
    fill_with_most_frequent(data, 'tip_posla')
    data = normalize_other_atr(data, 'tip_posla', tip_posla)
    data = smarter_version_one_hot(data, 'rasa')
    return data


def preprocess_test_dataframe(data, normalization_map):
    data = normalize_dataframe(data, normalization_map)
    data = normalize_other_atr(data, 'tip_posla', tip_posla)
    data = normalize_other_atr(data, 'zdravstveno_osiguranje', zdravstveno_osiguranje)
    data = smarter_version_one_hot(data, 'bracni_status')
    data = smarter_version_one_hot(data, 'rasa')
    return data


def random_forest(data, test):
    x_train = data.drop(columns=['obrazovanje'])
    y_train = data['obrazovanje']
    x_test = test.drop(columns=['obrazovanje'])
    y_test = test['obrazovanje']
    rf_classifier = RandomForestClassifier()
    rf_classifier.fit(x_train, y_train)
    y_pred = rf_classifier.predict(x_test)
    score = f1_score(y_test, y_pred, average='macro')
    print(score)


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
    train_data = add_rows(train_data)
    normalization_map = create_normalization_map(train_data)
    train_data = preprocess_dataframe(train_data)
    test_data = load_file(sys.argv[2])
    test_data = preprocess_test_dataframe(test_data, normalization_map)
    test_data = add_missing_columns(test_data, train_data.columns)
    random_forest(train_data, test_data)
