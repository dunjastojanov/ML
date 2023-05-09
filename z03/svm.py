import sys

import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC

STOPWORDS = ['koja', 'tim', 'onakve', 'tome', 'onolikim', 'si', 'ovolikom', 'onakvim', 'nekog', 'gde', 'i', 'takvim',
             'onoliki', 'onakav', 'ovoliku', 'onakvu', 'ovakvim', 'kojeg', 'ovakvo', 'onolikom', 'neke', 'o',
             'onolikoj', 'bio', 'kuda', 'ovolika', 'na', 'one', 'onakvoj', 'tolikim', 'ka', 'bilo', 'kakve', 'skoro',
             'odakle', 'onakvom', 'za', 'koliki', 'kolike', 'da', 'ako', 'kojom', 'te', 'tolike', 'onoliku', 'jednom',
             'odmah', 'tolikom', 'neka', 'takvu', 'ali', 'tolikog', 'onoliko', 'kojim', 'tom', 'koji', 'smo', 'ovu',
             'takav', 'kakvom', 'čim', 'šta', 'sa', 'što', 'ovoliko', 'jednog', 'onoj', 'ko', 'kako', 'spreman',
             'bila', 'još', 'ovo', 'takvoj', 'jesam', 'koga', 'u', 'to', 'ovakav', 'biti', 'zašto', 'ste', 'sam',
             'toliki', 'kod', 'onim', 'ona', 'jednoj', 'ono', 'ovakvog', 'chim', 'tolika', 'ovom', 'ovim', 'kad',
             'takvog', 'kakav', 'do', 'jednim', 'bas', 'sta', 'ja', 'takodje', 'jedne', 'ovaj', 'od', 'jedno',
             'ovoliki', 'onog', 'kakva', 'kakvim', 'štaviše', 'takvo', 'upravo', 'koliko', 'bezmalo', 'jedna',
             'onaj', 'ovakve', 'onolikog', 'pri', 'onakvo', 'sada', 'ovakvoj', 'ovolikim', 'onakva', 'kojoj', 'ove',
             'onolike', 'baš', 'koju', 'ovolikog', 'ovolike', 'neki', 'gotovo', 'oko', 'kakvoj', 'takve', 'onolika',
             'ovakvom', 'nekim', 'tu', 'odmakh', 'sto', 'nekom', 'pored', 'tolikoj', 'takode', 'zbog', 'kakvo', 'ova',
             'kakvog', 'jedan', 'taj', 'jesi', 'kojem', 'toliko', 'ovog', 'jos', 'toj', 'ili', 'jesmo', 'je',
             'ovolikoj', 'kao', 'a', 'koje', 'onakvog', 'jesu', 'onom', 'su', 'tog', 'jeste', 'ovakva', 'zasto',
             'ovoj', 'jednu', 'kakvu', 'kolika', 'jer', 'takvom', 'toliku', 'iz', 'nekoj', 'ta', 'otkuda', 'onu',
             'kada', 'takva', 'neko', 'se', 'ja', 'ti', 'on', 'ona', 'ono', 'mi', 'vi', 'oni', 'one', 'vam', 'ga', 'im',
             'ko', 'šta', "što", "nešto", "svako", "ih", "al", "bi"]

EMOTICONS = {
    ":)": "smeh",
    ":(": "tuga",
    ";)": "namig",
    ":D": "smeh",
    ":-)": "smeh",
    ":-(": "tuga",
    ";-)": "namig",
    ":'(": "tuga",
    ":')": "radost",
    ":|(": "iznenađenje",
    ":|)": "iznenađenje",
    ":o)": "šok",
    ":o(": "šok",
    ":p": "jezik",
    ":P": "jezik",
    ":d": "jezik",
    ":))": "smeh",
    ":)))": "smeh",
    ":((": "tuga",
    ":(((": "tuga",
}


def lowercase_review(review):
    return review.lower()


def remove_stop_words(review):
    return " ".join([word for word in review.split() if word not in STOPWORDS])


def remove_punctuation(review):
    return "".join([char for char in review if char not in """!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""])


def replace_emoticons(review):
    return " ".join([EMOTICONS[word] if word in EMOTICONS.keys() else word for word in review.split()])


def transform_zeros(sentiment):
    if sentiment == 0:
        return -1
    return sentiment


def transform_dataset(dataset):
    dataset['Review'] = dataset['Review'].apply(lowercase_review)
    dataset['Review'] = dataset['Review'].apply(replace_emoticons)
    dataset['Review'] = dataset['Review'].apply(remove_punctuation)
    dataset['Review'] = dataset['Review'].apply(remove_stop_words)
    dataset['Sentiment'] = dataset['Sentiment'].apply(transform_zeros)
    return dataset


def split_dataset_on_x_y(dataset):
    return dataset.loc[:, 'Review'], dataset.loc[:, 'Sentiment']


def linear_kernel(x_train, y_train, x_test, y_test):
    clf = SVC(kernel='linear')
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    accuracy = sklearn.metrics.f1_score(y_test, y_pred, average='micro')
    print(accuracy)


if __name__ == '__main__':
    train_data = pd.read_csv(sys.argv[1], sep='\t')
    test_data = pd.read_csv(sys.argv[2], sep='\t')
    train_data = transform_dataset(train_data)
    test_data = transform_dataset(test_data)
    x_train, y_train = split_dataset_on_x_y(train_data)
    x_test, y_test = split_dataset_on_x_y(test_data)
    vectorizer = TfidfVectorizer()
    x_train = vectorizer.fit_transform(x_train)
    x_test = vectorizer.transform(x_test)
    linear_kernel(x_train, y_train, x_test, y_test)
