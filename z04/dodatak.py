# def boosting(data, test_data):
#     x_train = data.drop(columns=['obrazovanje'])
#     y_train = data['obrazovanje']
#     x_test = test_data.drop(columns=['obrazovanje'])
#     y_test = test_data['obrazovanje']
#     gb_classifier = AdaBoostClassifier()
#     gb_classifier.fit(x_train, y_train)
#     y_pred = gb_classifier.predict(x_test)
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

