import sys

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sb
from sklearn.metrics import v_measure_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split

more = {
    "da": 1,
    "ne": 0
}


def correlation_matrix(data):
    matrica_korelacije = data.corr()
    plt.figure(figsize=(15, 15))
    sb.heatmap(matrica_korelacije, cmap="Blues", annot=True)
    plt.show()


def histogram(data, atr_name):
    sb.distplot(data[atr_name], bins=40, axlabel=atr_name)
    plt.show()


def box_plot(data, atr_name):
    plt.boxplot(data[atr_name])
    plt.title(atr_name)
    plt.show()


def load_file(file_name):
    return pd.read_csv(file_name)


def remove_row_with_nan_atr(data, count_of_nan_atr=2):
    mask = data.isna().sum(axis=1) > count_of_nan_atr
    return data.drop(data[mask].index)


def remove_row_for_value(data, atr_name, value):
    data = data.drop(data[(data[atr_name] > value)].index)
    return data


def smarter_version_one_hot(data, atr_name):
    df_dummy = pd.get_dummies(data[atr_name], prefix=atr_name).iloc[:, :-1]
    new_df = pd.concat([data, df_dummy], axis=1)
    new_df.drop([atr_name], axis=1, inplace=True)
    return new_df


def fill_in_izvoz(data):
    median_izvoz = data['Izvoz'].median()
    data['Izvoz'] = data['Izvoz'].fillna(median_izvoz)
    return data


def fill_in_BDP(data):
    median_izvoz = data['BDP'].mean()
    data['BDP'] = data['BDP'].fillna(median_izvoz)
    return data


def normalize_other_atr(data, atr_name, map):
    data[atr_name] = data[atr_name].map(map)
    return data


def drop_column_with_name(data, name):
    data = data.drop(name, axis=1)
    return data


def preprocess_dataframe(data):
    data = remove_row_with_nan_atr(data, count_of_nan_atr=2)
    data = fill_in_BDP(data)
    data = fill_in_izvoz(data)
    box_plot(data, 'Izvoz')
    box_plot(data, 'Inflacija')
    box_plot(data, 'BDP')
    data = remove_row_for_value(data, 'Izvoz', 125)
    data = remove_row_for_value(data, 'Inflacija', 500)
    data = remove_row_for_value(data, 'BDP', 150000)
    data = normalize_other_atr(data, 'More', more)
    data = smarter_version_one_hot(data, 'Religija')
    return data


def preprocess_test_dataframe(data):
    data = normalize_other_atr(data, 'More', more)
    data = smarter_version_one_hot(data, 'Religija')
    return data


def add_missing_columns(test, column_names):
    dataframe = pd.DataFrame(columns=column_names)
    for name in column_names:
        if name not in test.columns:
            dataframe[name] = pd.Series(np.zeros(test.shape[0]))
        else:
            dataframe[name] = test[name]
    return dataframe


def EM(train_data):
    num_clusters = 4
    y = train_data['Region']
    data = drop_column_with_name(train_data, 'Region')
    X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.3)
    gmm = GaussianMixture(n_components=num_clusters, covariance_type='spherical', init_params='kmeans', random_state=42)
    gmm.fit(X_train)
    predicted_clusters = gmm.predict(X_test)
    print('spherical ', v_measure_score(y_test, predicted_clusters))
    gmm = GaussianMixture(n_components=num_clusters, covariance_type='full', init_params='kmeans', random_state=42)
    gmm.fit(X_train)
    predicted_clusters = gmm.predict(X_test)
    print('full ', v_measure_score(y_test, predicted_clusters))
    gmm = GaussianMixture(n_components=num_clusters, covariance_type='tied', init_params='kmeans', random_state=42)
    gmm.fit(X_train)
    predicted_clusters = gmm.predict(X_test)
    print('tied ', v_measure_score(y_test, predicted_clusters))
    gmm = GaussianMixture(n_components=num_clusters, covariance_type='diag', init_params='kmeans', random_state=42)
    gmm.fit(X_train)
    predicted_clusters = gmm.predict(X_test)
    print('diag ', v_measure_score(y_test, predicted_clusters))


def gaussian(test_data, train_data):
    num_clusters = 4
    x_train = drop_column_with_name(train_data, 'Region')
    y_test = test_data['Region']
    x_test = drop_column_with_name(test_data, 'Region')
    gmm = GaussianMixture(n_components=num_clusters, covariance_type='diag', init_params='kmeans', random_state=42)
    gmm.fit(x_train)
    predicted_clusters = gmm.predict(x_test)
    print(v_measure_score(y_test, predicted_clusters))


if __name__ == '__main__':
    train_data = load_file('train.csv')
    train_data = preprocess_dataframe(train_data)

    test_data = load_file('test_preview.csv')
    test_data = preprocess_test_dataframe(test_data)

    gaussian(test_data, train_data)

    # samo izvoz treba da namapiramo
    # print(train_data.info())
    # correlation_matrix(train_data)
    # print(train_data.cov())

    EM(train_data)
