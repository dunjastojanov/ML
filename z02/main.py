import sys

import numpy as np
import pandas


def load_file(file_path):
    return pandas.read_csv(file_path)


def implement_one_hot_encoding(dataframe):
    one_hot_encoded = pandas.get_dummies(dataframe[['make', 'category', 'fuel']])
    return pandas.concat([dataframe, one_hot_encoded], axis=1)


def drop_columns(dataframe):
    dataframe = dataframe.drop("color", axis=1)
    dataframe = dataframe.drop("transmission", axis=1)
    dataframe = dataframe.drop("make", axis=1)
    dataframe = dataframe.drop("category", axis=1)
    return dataframe.drop("fuel", axis=1)


def create_normalization_map(dataframe):
    map = {}
    for column in dataframe.columns:
        if column in ['mileage', 'price', 'engine_size', 'year']:
            map[column] = {
                'min': dataframe[column].min(),
                'max': dataframe[column].max()
            }
    return map


def kernel(distance, lamda):
    return (3 / (4 * lamda)) * (1 - ((distance / lamda) ** 2))


def knn(train, test, k, lamda):
    row_number = test.shape[0]
    predictions = np.zeros(row_number)
    for i in range(row_number):
        distances = np.sqrt(np.sum((train.iloc[:, :-1].values - test.iloc[i, :-1].values) ** 2, axis=1))
        k_nearest_neighbours = np.argsort(distances)[:k]
        weights = np.zeros(k)
        for j in range(k):
            distance = distances[k_nearest_neighbours[j]]
            weights[j] = kernel(distance, lamda)
        weights /= np.sum(weights)
        k_nearest_classes = train.iloc[k_nearest_neighbours, -1]
        predictions[i] = np.sum(weights * k_nearest_classes)
    return predictions


def rmse(y_values, calculated_values):
    suma = 0
    for expected, actual in zip(calculated_values, y_values):
        suma += (actual - expected) ** 2
    return np.sqrt(suma / len(y_values))


def remove_faulty_data(data):
    for row in range(len(data['price'])):
        if data['price'][row] == 0:
            data.drop(row)
    return data


def permute_columns(dataframe):
    return pandas.concat([dataframe.iloc[:, 4:], dataframe.iloc[:, :4]], axis=1)


def normalize_dataframe(dataframe, map_for_normalization):
    for column in map_for_normalization.keys():
        dataframe[column] = (dataframe[column] - map_for_normalization[column]["min"]) / (
                map_for_normalization[column]["max"] - map_for_normalization[column]["min"])
    return dataframe


def denormalize_y(column, map_for_normalization):
    column = column * (
            map_for_normalization["price"]["max"] - map_for_normalization["price"]["min"]) + \
             map_for_normalization["price"]["min"]
    return column


def add_missing_columns(test_data, column_names):
    dataframe = pandas.DataFrame(columns=column_names)
    for name in column_names:
        if name not in test_data.columns:
            dataframe[name] = pandas.Series(np.zeros(test_data.shape[0]))
        else:
            dataframe[name] = test_data[name]
    return dataframe


def preprocess_data(file_path, type='train'):
    data = load_file(file_path)
    if type == 'train':
        data = remove_faulty_data(data)
    data = implement_one_hot_encoding(data)
    data = drop_columns(data)
    return permute_columns(data)


if __name__ == '__main__':
    train_data = preprocess_data(sys.argv[1])
    test_data = preprocess_data(sys.argv[2], type='test')
    expected_values = test_data['price']
    normalizing_map = create_normalization_map(train_data)
    train_data = normalize_dataframe(train_data, normalizing_map)
    test_data = normalize_dataframe(test_data, normalizing_map)
    test_data = add_missing_columns(test_data, train_data.columns)
    predictions = knn(train_data, test_data, 5, lamda=3)
    predictions = denormalize_y(predictions, normalizing_map)
    print(rmse(expected_values, predictions))
