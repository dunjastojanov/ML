import string

import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


def transform_review_sentence(review):
    review.translate(str.maketrans('', '', string.punctuation))
    return review.lower()


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


def remove_stop_words(review):
    return " ".join([word for word in review.split() if word not in STOPWORDS])


def remove_punctuation(review):
    return "".join([char for char in review if char not in string.punctuation])


def replace_emoticons(review):
    return " ".join([EMOTICONS[word] if word in EMOTICONS.keys() else word for word in review.split()])


def transform_dataset(dataset):
    dataset['Review'] = dataset['Review'].apply(transform_review_sentence)
    dataset['Review'] = dataset['Review'].apply(replace_emoticons)
    dataset['Review'] = dataset['Review'].apply(remove_punctuation)
    dataset['Review'] = dataset['Review'].apply(remove_stop_words)
    dataset['Sentiment'] = dataset['Sentiment'].apply(transform_zeros)
    return dataset


def transform_zeros(sentiment):
    if sentiment == 0:
        return -1
    return sentiment


def split_dataset_on_x_y(dataset):
    return dataset.loc[:, 'Review'], dataset.loc[:, 'Sentiment']


def linear_kernel():
    clf = SVC(kernel='linear')
    clf.fit(X_train, y_train)
    # Evaluate the classifier on the testing data
    y_pred = clf.predict(X_test)
    accuracy = sklearn.metrics.f1_score(y_test, y_pred, average='micro')
    # accuracy = accuracy_score(y_test, y_pred)
    print(accuracy)
    return accuracy


if __name__ == '__main__':
    # Load the dataset

    train_data = pd.read_csv('train.tsv', sep='\t')
    train_data = transform_dataset(train_data)
    # test_data = pd.read_csv('test_preview.tsv', sep='\t')
    # transform_dataset(test_data)
    # X_test, y_test = split_dataset_on_x_y(test_data)
    # X_train, y_train = split_dataset_on_x_y(train_data)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(train_data['Review'], train_data['Sentiment'], test_size=0.2)

    # Convert the text reviews into numerical features using TF-IDF vectorization
    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)

    # Train an SVM classifier on the training data
    linear_kernel()
