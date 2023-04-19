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


def replace_emojis(review):
    if ':)' in review:
        review = review.replace(":)", "dobro")
    if ';)' in review:
        review = review.replace(";)", "super")
    if 'xD' in review:
        review = review.replace("xD", "fantasticno")
    if ':-)' in review:
        review = review.replace(":-)", "dobro")
    if ';D' in review:
        review = review.replace(";D", "bez veze")
    if ':(' in review:
        review = review.replace(":(", "lose")
    return review


def remove_stop_words(review):
    # TODO implement removing stop words
    pass


def remove_serbian_specific_letters(review):
    # TODO implement replacing letters like ŠĐĆČŽ with s,dj,c,c,z
    pass


def transform_zeros(sentiment):
    if sentiment == 0:
        return -1
    return sentiment


def transform_dataset(dataset):
    dataset['Review'] = dataset['Review'].apply(replace_emojis)
    dataset['Review'] = dataset['Review'].apply(transform_review_sentence)
    # TODO uncomment when done
    # dataset['Review'] = dataset['Review'].apply(remove_serbian_specific_letters)
    # dataset['Review'] = dataset['Review'].apply(remove_stop_words)
    dataset['Sentiment'] = dataset['Sentiment'].apply(transform_zeros)
    return dataset


def split_dataset_on_x_y(dataset):
    return dataset.loc[:, 'Review'], dataset.loc[:, 'Sentiment']


def linear_kernel():
    clf = SVC(kernel='linear')
    clf.fit(X_train, y_train)
    # Evaluate the classifier on the testing data
    y_pred = clf.predict(X_test)
    accuracy = sklearn.metrics.f1_score(y_test, y_pred, average='micro')
    # accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
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
