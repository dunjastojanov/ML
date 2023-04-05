import pandas


def load_file(file_path):
    return pandas.read_csv(file_path)


def implement_one_hot_encoding(dataframe):
    one_hot_encoded = pandas.get_dummies(dataframe[['make', 'category', 'year', 'fuel']])
    return pandas.concat([dataframe, one_hot_encoded], axis=1)


def drop_columns(dataframe):
    dataframe = dataframe.drop("color", axis=1)
    dataframe = dataframe.drop("transmission", axis=1)
    dataframe = dataframe.drop("make", axis=1)
    dataframe = dataframe.drop("category", axis=1)
    dataframe = dataframe.drop("fuel", axis=1)
    return dataframe.drop("year", axis=1)


def create_map_with_max_and_min_column_value(dataframe):
    map = {}
    counter = 0
    for column in dataframe.columns:
        map[counter] = {
            'min': dataframe[column].min(),
            'max': dataframe[column].max()
        }
        counter += 1
        if counter >= 3:
            break
    return map


def normalize_dataframe(dataframe, map_for_normalization):
    counter = 0
    for column in dataframe.columns:
        dataframe[column] = (dataframe[column] - map_for_normalization[counter]["min"]) / (
                map_for_normalization[counter]["max"] - map_for_normalization[counter]["min"])
        counter += 1
        if counter >= 3:
            break
    return dataframe


def denormalize_dataframe(dataframe, map_for_normalization):
    counter = 0
    for column in dataframe.columns:
        dataframe[column] = dataframe[column] * (
                map_for_normalization[counter]["max"] - map_for_normalization[counter]["min"]) + \
                            map_for_normalization[counter]["min"]
        counter += 1
        if counter >= 3:
            break
    return dataframe


if __name__ == '__main__':
    df = load_file("./train.csv")
    print(df)
    df = implement_one_hot_encoding(df)
    df = drop_columns(df)
    print(df)
    normalizing_map = create_map_with_max_and_min_column_value(df)
    df = normalize_dataframe(df, normalizing_map)
    print(df)
