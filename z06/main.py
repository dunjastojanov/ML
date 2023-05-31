import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.decomposition import PCA
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
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
    median_izvoz = data[atr_name].median()
    data[atr_name] = data[atr_name].fillna(median_izvoz)
    return data


def fill_with_most_frequent(data, atr_name):
    most_frequent_atr = data[atr_name].mode()[0]
    data[atr_name].fillna(most_frequent_atr, inplace=True)
    return data


def normalize_other_atr(data, atr_name, map):
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
    data = normalize_other_atr(data, 'jastuk', da_ne)
    data = normalize_other_atr(data, 'pojas', da_ne)
    data = normalize_other_atr(data, 'pol', pol)
    data = normalize_other_atr(data, 'tip', tip)
    return data


def preprocess_test_dataframe(data):
    return data


def add_missing_columns(test, column_names):
    dataframe = pd.DataFrame(columns=column_names)
    for name in column_names:
        if name not in test.columns:
            dataframe[name] = pd.Series(np.zeros(test.shape[0]))
        else:
            dataframe[name] = test[name]
    return dataframe


def algo(train_data, test_data):
    y_train = train_data['ishod']
    train_data = drop_column_with_name(train_data, 'ishod')
    pca = PCA(n_components=5)  # n_components=None za prikaz svega
    pca.fit(train_data)
    component = pca.transform(train_data)
    # x_test = test_data.drop(columns=['obrazovanje'])
    # y_test = test_data['obrazovanje']
    x_train, x_test, y_train, y_test = train_test_split(component, y_train, test_size=0.2, random_state=42,
                                                        stratify=y_train)
    tree = DecisionTreeClassifier(max_depth=7)
    adaboost = AdaBoostClassifier(base_estimator=tree, n_estimators=100, random_state=42)
    adaboost.fit(x_train, y_train)
    y_pred = adaboost.predict(x_test)
    score = f1_score(y_test, y_pred, average='macro')
    print(score)


if __name__ == '__main__':
    train_data = pd.read_csv('train.csv')
    train_data = preprocess_dataframe(train_data)
    algo(train_data, train_data)
    # pca = PCA(n_components=3)  # n_components=None za prikaz svega
    # pca.fit(drop_column_with_name(train_data, 'ishod'))
    # plt.plot(np.cumsum(pca.explained_variance_ratio_))
    # plt.show()
