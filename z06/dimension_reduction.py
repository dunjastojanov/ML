import sys

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier

da_ne = {
    'da': 1,
    'ne': 0
}
pol = {
    'm': 0,
    'z': 1
}
tip = {
    'vozac': 0,
    'putnik': 1
}


def load_file(file_name):
    return pd.read_csv(file_name)


def drop_column_with_name(data, name):
    data = data.drop(name, axis=1)
    return data


def remove_row_with_nan_atr(data, count_of_nan_atr=2):
    mask = data.isna().sum(axis=1) > count_of_nan_atr
    return data.drop(data[mask].index)


def remove_outlier_where_y_is_nan(data):
    return data.drop(data[(data['ishod'].isna())].index)


def remove_row_for_value_higher(data, atr_name, value):
    data = data.drop(data[(data[atr_name] > value)].index)
    return data


def remove_row_for_value_less(data, atr_name, value):
    data = data.drop(data[(data[atr_name] < value)].index)
    return data


def fill_in_with_median(data, atr_name):
    median = data[atr_name].median()
    data[atr_name] = data[atr_name].fillna(median)
    return data


def fill_with_most_frequent(data, atr_name):
    most_frequent_atr = data[atr_name].mode()[0]
    data[atr_name].fillna(most_frequent_atr, inplace=True)
    return data


def normalize_with_map(data, atr_name, map):
    data[atr_name] = data[atr_name].map(map)
    return data


def preprocess_dataframe(data):
    data = remove_outlier_where_y_is_nan(data)
    data = remove_row_for_value_higher(data, 'masa', 15000)
    data = remove_row_for_value_less(data, 'godina', 1967)
    data = fill_in_with_median(data, 'starost')
    data = fill_in_with_median(data, 'godina')
    data = fill_in_with_median(data, 'masa')
    data = fill_with_most_frequent(data, 'jastuk')
    data = fill_with_most_frequent(data, 'pojas')
    data = fill_with_most_frequent(data, 'ceoni')
    data = fill_with_most_frequent(data, 'pol')
    data = fill_with_most_frequent(data, 'tip')
    data = normalize_with_map(data, 'jastuk', da_ne)
    data = normalize_with_map(data, 'pojas', da_ne)
    data = normalize_with_map(data, 'pol', pol)
    data = normalize_with_map(data, 'tip', tip)
    return data


def preprocess_test_dataframe(data):
    data = normalize_with_map(data, 'jastuk', da_ne)
    data = normalize_with_map(data, 'pojas', da_ne)
    data = normalize_with_map(data, 'pol', pol)
    data = normalize_with_map(data, 'tip', tip)
    return data


def algo(train_data, test_data):
    y_train = train_data['ishod']
    train_data = drop_column_with_name(train_data, 'ishod')
    pca = PCA(n_components=3)
    pca.fit(train_data)
    x_train = pca.transform(train_data)
    x_test = pca.transform(test_data.drop(columns=['ishod']))
    y_test = test_data['ishod']
    tree = DecisionTreeClassifier(max_depth=7)
    adaboost = AdaBoostClassifier(base_estimator=tree, n_estimators=100, random_state=42)
    adaboost.fit(x_train, y_train)
    y_pred = adaboost.predict(x_test)
    score = f1_score(y_test, y_pred, average='macro')
    print(score)


if __name__ == '__main__':
    train_data = pd.read_csv(sys.argv[1])
    test_data = pd.read_csv(sys.argv[2])
    train_data = preprocess_dataframe(train_data)
    test_data = preprocess_test_dataframe(test_data)
    algo(train_data, test_data)
