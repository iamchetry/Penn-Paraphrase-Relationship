from phrase_relations.constants import *

from re import *
from pandas import *
from collections import Counter


def load_data(file_):
    data_ = read_csv('data/{}'.format(file_), header=None).dropna(axis=1)
    data_.columns = range(len(data_.columns))
    return data_.rename(columns={len(data_.columns)-1: 'label'})


def identify_nums(str_):
    try:
        return float(str_)
    except ValueError as e:
        print e.message


def list_to_dict(list_):
    iter_ = iter(list_)
    return dict(zip(iter_, iter_))


def replace_missing_values(data_):
    continuous_cols = list(set(data_.columns).difference(set(categorical_columns)))
    data_[continuous_cols] = data_[continuous_cols].apply(to_numeric, errors='coerce')
    data_[continuous_cols] = data_[continuous_cols].fillna(data_[continuous_cols].mean())

    return data_


def remove_cols(data_):
    return data_[[col_ for col_ in data_.columns if len(data_[col_].unique()) > 1]]


def split_columns(data_):
    len_ = len(data_.columns)
    for col_ in range(3, len_-2):
        data_ = concat([data_.drop([col_], axis=1), data_[col_].apply(Series)], axis=1)

    return data_


def fix_duplicate_cols(data_):
    col_dict = Counter(data_.columns)
    duplicate_cols = [key_ for key_ in col_dict if col_dict[key_] > 1]
    for col_ in duplicate_cols:
        data_ = concat([data_.drop([col_], axis=1), DataFrame(data_[col_].values, columns=[
            str(col_)+'_'+str(k) for k in range(len(data_[col_].columns))])], axis=1)

    return data_


def extract_values(data_):
    data_[range(3, len(data_.columns)-2)] = data_[range(3, len(data_.columns)-2)].astype('str').\
        applymap(lambda str_: [k for k in split('=| ', str_) if k != ''])
    data_[range(3, len(data_.columns)-2)] = data_[range(3, len(data_.columns)-2)].\
        applymap(lambda list_: list_to_dict(list_))
    data_ = split_columns(data_)
    data_.columns = [sub('[,)(]', '', str(col_)) for col_ in data_.columns]

    return data_

