import pandas as pd


def load_file(file_name):
    return pd.read_csv(file_name)


def remove_row_with_nan_atr(train_data, count_of_nan_atr=2):
    mask = train_data.isna().sum(axis=1) > count_of_nan_atr
    return train_data.drop(train_data[mask].index)




if __name__ == '__main__':
    train_data = load_file('train.csv')
    df = train_data[(train_data.isna().sum(axis=1) > 1)]
    print(len(df))
    print(len(train_data))
    print(len(remove_row_with_nan_atr(train_data)))
