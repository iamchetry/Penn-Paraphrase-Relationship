from phrase_relations.constants import *

from re import *
from pandas import *
from collections import Counter


def load_data(file_, label_col=None):
    '''
    :param file_: csv file to load
    :param label_col: class label column name
    :return: data-frame
    '''

    data_ = read_csv('data/{}'.format(file_), header=None).dropna(axis=1)
    data_.columns = range(len(data_.columns))
    return data_.rename(columns={len(data_.columns) - 1: label_col, 1: 'source', 2: 'target'})


def list_to_dict(list_):
    '''
    :param list_: list to convert into a dictionary with its 2 successive terms as a (key, value) pair
    :return: dictionary
    '''

    iter_ = iter(list_)
    return dict(zip(iter_, iter_))


def replace_missing_values(data_):
    '''
    :param data_: data-frame whose missing values to be filled
    :return: data-frame
    '''

    continuous_cols = list(set(data_.columns).difference(set(categorical_columns)))
    data_[continuous_cols] = data_[continuous_cols].apply(to_numeric, errors='coerce')
    data_[continuous_cols] = data_[continuous_cols].fillna(data_[continuous_cols].mean())

    return data_


def remove_cols(data_):
    '''
    :param data_: data-frame
    :return: list of columns
    '''

    return [col_ for col_ in data_.columns if len(data_[col_].unique()) > 1]


def split_columns(data_):
    '''
    :param data_: data-frame whose certain columns split to multiple columns
    :return: data-frame
    '''

    len_ = len(data_.columns)
    for col_ in range(3, len_ - 2):
        data_ = concat([data_.drop([col_], axis=1), data_[col_].apply(Series)], axis=1)

    return data_


def fix_duplicate_cols(data_):
    '''
    :param data_: data-frame with duplicate columns
    :return: data-frame with no duplicate column
    '''

    col_dict = Counter(data_.columns)
    duplicate_cols = [key_ for key_ in col_dict if col_dict[key_] > 1]
    for col_ in duplicate_cols:
        data_ = concat([data_.drop([col_], axis=1), DataFrame(data_[col_].values, columns=[
            str(col_) + '_' + str(k) for k in range(len(data_[col_].columns))])], axis=1)

    return data_


def extract_values(data_):
    '''
    :param data_: data-frame to extract numeric values from some of its columns
    :return: data-frame
    '''

    data_[range(3, len(data_.columns) - 2)] = data_[range(3, len(data_.columns) - 2)].astype('str'). \
        applymap(lambda x: [k for k in split('=| ', x) if k != ''])
    data_[range(3, len(data_.columns) - 2)] = data_[range(3, len(data_.columns) - 2)]. \
        applymap(lambda list_: list_to_dict(list_))
    data_ = split_columns(data_)
    data_.columns = [sub('[,)(]', '', str(col_)) for col_ in data_.columns]

    return data_


def clean_text(text_):
    '''
    :param text_: string/text
    :return: refined string/text
    '''

    return sub('[^a-z0-9A-Z]+', ' ', text_).lower()


def tokenize_phrase(str_):
    '''
    :param str_: string phrase
    :return: list of tokenized words
    '''

    return [word_ for word_ in str_.split() if word_ != '']


def take_common_cols(data_1, data_2):
    '''
    :param data_1: first data-frame
    :param data_2: second data-frame
    :return: list of common columns
    '''

    return list(set(data_1.columns).intersection(set(data_2.columns)))


def calculate_mean_of_vecs(list_of_lists, dim_=None):
    '''
    :param list_of_lists: list of vectors
    :param dim_: dimension of vectors
    :return: list
    '''

    sum_vec = np.zeros(dim_)
    for _, val_ in enumerate(list_of_lists):
        sum_vec = sum_vec + np.array(val_)

    return list(sum_vec/len(list_of_lists))

