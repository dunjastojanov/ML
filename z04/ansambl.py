import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

minium_years = {
    "osnovne studije": 20.0,
    "master": 26.0,
    "doktorat": 29.0,
    "srednja skola": 16.0
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


def histogram(data, atr_name):
    # zbog ovoga uzimamo mean za godine normalna raspodela
    # zbog ovoga uzimamo medianu za plate -> nije normalna raspodela
    sb.distplot(data[atr_name], bins=40, axlabel=atr_name)
    plt.show()


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


def preprocess_dataframe(data):
    data = remove_row_with_nan_atr(data)
    data = remove_outlier_where_y_is_nan(data)
    data = fill_in_age(data)
    data = remove_outlier_by_age(data)
    data = fill_in_salary(data)
    # data = drop_column_with_name(data, 'tip_posla')
    # data = drop_column_with_name(data, 'bracni_status')
    # data = drop_column_with_name(data, 'zdravstveno_osiguranje')
    # data = drop_column_with_name(data, 'rasa')
    normalization_map = create_normalization_map(data)
    data = normalize_dataframe(data, normalization_map)
    fill_with_most_frequent(data, 'rasa')
    fill_with_most_frequent(data, 'bracni_status')
    fill_with_most_frequent(data, 'tip_posla')
    fill_with_most_frequent(data, 'zdravstveno_osiguranje')
    data = normalize_other_atr(data, 'tip_posla', tip_posla)
    data = normalize_other_atr(data, 'zdravstveno_osiguranje', zdravstveno_osiguranje)
    data = smarter_version_one_hot(data, 'bracni_status')
    data = smarter_version_one_hot(data, 'rasa')
    return data


def preprocess_test_dataframe(data, normalization_map):
    data = normalize_dataframe(data, normalization_map)
    data = normalize_other_atr(data, 'tip_posla', tip_posla)
    data = normalize_other_atr(data, 'zdravstveno_osiguranje', zdravstveno_osiguranje)
    data = smarter_version_one_hot(data, 'bracni_status')
    data = smarter_version_one_hot(data, 'rasa')
    return data


def correlation_matrix(data):
    matrica_korelacije = data.corr()
    plt.figure(figsize=(15, 15))
    sb.heatmap(matrica_korelacije, cmap="Blues", annot=True)
    plt.show()


def bagging_algo(data):
    X = data.drop(columns=['obrazovanje'])
    y = data['obrazovanje']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    # Create a decision tree classifier
    tree = DecisionTreeClassifier(random_state=42)
    # Create a bagging ensemble of decision tree classifiers
    bagging = BaggingClassifier(base_estimator=tree, n_estimators=10, random_state=42)
    # Train the bagging ensemble on the training set
    bagging.fit(X_train, y_train)
    # Evaluate the performance of the bagging ensemble on the testing set
    y_pred = bagging.predict(X_test)
    score = f1_score(y_test, y_pred, average='macro')
    print("Macro F1 score:", score)


def boosting(data):
    y = data['obrazovanje']
    X = data.drop(columns=['obrazovanje'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    # Create a gradient boosting classifier object
    gb_classifier = AdaBoostClassifier()

    # Fit the model on the training data
    gb_classifier.fit(X_train, y_train)
    # Use the model to make predictions on the test data
    y_pred = gb_classifier.predict(X_test)
    # Calculate the macro f1 score
    score = f1_score(y_test, y_pred, average='macro')
    # Print the classification report and macro f1 score
    print("Macro F1 score:", score)


def random_forest(data):
    y = data['obrazovanje']
    X = data.drop(columns=['obrazovanje'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    # Create a random forest classifier object
    rf_classifier = RandomForestClassifier()
    # Fit the model on the training data
    rf_classifier.fit(X_train, y_train)
    # Use the model to make predictions on the test data
    y_pred = rf_classifier.predict(X_test)
    # Calculate the macro f1 score
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    # Print the classification report and macro f1 score
    print("Macro F1 score:", macro_f1)


if __name__ == '__main__':
    train_data = load_file('train.csv')
    train_data = preprocess_dataframe(train_data)
    bagging_algo(train_data)
    boosting(train_data)
    random_forest(train_data)
