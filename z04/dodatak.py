# def boosting(data, test_data):
#     x_train = data.drop(columns=['obrazovanje'])
#     y_train = data['obrazovanje']
#     x_test = test_data.drop(columns=['obrazovanje'])
#     y_test = test_data['obrazovanje']
# tree = DecisionTreeClassifier(max_depth=4)
# adaboost = AdaBoostClassifier(base_estimator=tree, n_estimators=100)
#     adaboost.fit(x_train, y_train)
#     y_pred = adaboost.predict(x_test)
#     score = f1_score(y_test, y_pred, average='macro')
#     print(score)
#
#
# def random_forest(data, test_data):
#     x_train = data.drop(columns=['obrazovanje'])
#     y_train = data['obrazovanje']
#     x_test = test_data.drop(columns=['obrazovanje'])
#     y_test = test_data['obrazovanje']
#     rf_classifier = RandomForestClassifier()
#     rf_classifier.fit(x_train, y_train)
#     y_pred = rf_classifier.predict(x_test)
#     macro_f1 = f1_score(y_test, y_pred, average='macro')
#     print(macro_f1)

# def bagging_algo(data, test_data):
#     x_train = data.drop(columns=['obrazovanje'])
#     y_train = data['obrazovanje']
#     x_test = test_data.drop(columns=['obrazovanje'])
#     y_test = test_data['obrazovanje']
#     # x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=42, stratify=y_train)
#     tree = DecisionTreeClassifier(random_state=42)
#     bagging = BaggingClassifier(base_estimator=tree, n_estimators=10, random_state=42)
#     bagging.fit(x_train, y_train)
#     y_pred = bagging.predict(x_test)
#     score = f1_score(y_test, y_pred, average='macro')
#     print(score)


# def correlation_matrix(data):
#     matrica_korelacije = data.corr()
#     plt.figure(figsize=(15, 15))
#     sb.heatmap(matrica_korelacije, cmap="Blues", annot=True)
#     plt.show()


# def histogram(data, atr_name):
#     # zbog ovoga uzimamo mean za godine normalna raspodela
#     # zbog ovoga uzimamo medianu za plate -> nije normalna raspodela
#     sb.distplot(data[atr_name], bins=40, axlabel=atr_name)
#     plt.show()

# def add_rows(dataframe, percentage=0.1):
#     for key in minium_years.keys():
#         filtered_df = dataframe[dataframe['obrazovanje'] == key]
#         new_row = {
#             'godine': filtered_df['godine'].median(),
#             'bracni_status': filtered_df['bracni_status'].mode()[0],
#             'rasa': filtered_df['rasa'].mode()[0],
#             'tip_posla': filtered_df['tip_posla'].mode()[0],
#             'zdravstveno_osiguranje': filtered_df['zdravstveno_osiguranje'].mode()[0],
#             'plata': filtered_df['plata'].median(),
#             'obrazovanje': key
#         }
#         stop = int(len(filtered_df) * percentage)
#         for i in range(stop):
#             dataframe.append(new_row, ignore_index=True)
#     return dataframe

def boosting_algo(data, test_data):
    param_grid = {
        'base_estimator__max_depth': [3, 5, 7],
        'n_estimators': [10, 20, 30]
    }
    x_train = data.drop(columns=['obrazovanje'])
    y_train = data['obrazovanje']
    # x_test = test_data.drop(columns=['obrazovanje'])
    # y_test = test_data['obrazovanje']
    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2,
                                                        stratify=y_train, random_state=42)
    # tree = DecisionTreeClassifier()
    # bagging = BaggingClassifier(base_estimator=tree)
    #
    # bagging.fit(x_train, y_train)
    # y_pred = bagging.predict(x_test)
    # score = f1_score(y_test, y_pred, average='macro')
    # print(score)
    # tree = DecisionTreeClassifier()
    # bagging = BaggingClassifier(base_estimator=tree)
    param_grid = {
        'n_estimators': [50, 100, 150],
        'base_estimator__max_depth': [5, 7, 10, 15]
    }
    # tree = DecisionTreeClassifier(max_depth=7)
    # adaboost = AdaBoostClassifier(base_estimator=tree, n_estimators=100, random_state=42)
    #
    # adaboost.fit(x_train, y_train)
    # y_pred = adaboost.predict(x_test)
    # score = f1_score(y_test, y_pred, average='macro')
    # print(score)
    # grid_search = GridSearchCV(estimator=adaboost, param_grid=param_grid, cv=5, scoring='f1_macro')
    #
    # grid_search.fit(x_train, y_train)
    #
    # print("Best parameters:", grid_search.best_params_)
    # print("Best score:", grid_search.best_score_)
