import sys

import pandas
import numpy as np


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


def knn_kernel(train, test, k, lamda=1):
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


def cross_validation(df, k=5):
    normalizing_map = create_normalization_map(df)
    df = normalize_dataframe(df, normalizing_map)
    rmse_scores = []

    # Set the random seed for reproducibility
    np.random.seed(42)

    # Shuffle the rows of the DataFrame
    df_shuffled = df.sample(frac=1).reset_index(drop=True)

    # Split the shuffled DataFrame into k parts
    k = 5
    df_parts = np.array_split(df_shuffled, k)

    # Concatenate all parts except one into a single DataFrame
    i = 0  # The index of the part to exclude

    for i in range(k):
        df_train = pandas.concat([df_parts[j] for j in range(k) if j != i])
        df_test = pandas.concat([df_parts[i]])

        # coef = ridge_regression(df_train, alpha=385)
        y_test = df_test['price']

        y_test_denormalized = denormalize_y(y_test, normalizing_map)
        # y_calculated_denormalized = denormalize_y(predict(df, coef), normalizing_map)
        # rmse_scores.append(rmse(y_test_denormalized, y_calculated_denormalized))

        predictions = knn_kernel(df_train, df_test, 10, lamda=10)
        predictions = denormalize_y(predictions, normalizing_map)
        rmse_scores.append(rmse(y_test_denormalized, predictions))

    mean_score = np.mean(rmse_scores)

    print(mean_score)
    print(rmse_scores)


def add_missing_columns(test_data, column_names):
    for name in column_names:
        if name not in test_data.columns:
            test_data = pandas.concat([pandas.Series(np.zeros(test_data.shape[0])), test_data], axis=1)
    return test_data.sort_index(axis=1, key=lambda x: pandas.Index(column_names).get_indexer(x))


def preprocess_data(file_path, type='train'):
    data = load_file(file_path)
    if type == 'train':
        data = remove_faulty_data(data)
    data = implement_one_hot_encoding(data)
    data = drop_columns(data)
    return permute_columns(data)


if __name__ == '__main__':
    train_data = preprocess_data(sys.argv[1])
    test_data = preprocess_data(sys.argv[2])
    expected_values = test_data['price']
    normalizing_map = create_normalization_map(train_data)
    train_data = normalize_dataframe(train_data, normalizing_map)
    test_data = normalize_dataframe(test_data, normalizing_map)
    test_data = add_missing_columns(test_data, train_data.columns)
    predictions = knn_kernel(train_data, test_data, 5, lamda=3)
    predictions = denormalize_y(predictions, normalizing_map)
    print(rmse(expected_values, predictions))
