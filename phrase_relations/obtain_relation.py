from phrase_relations.constants import *

from re import *
from pandas import *
from collections import Counter
from gensim.models import Word2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from imblearn.over_sampling import SMOTE

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale


def load_data(file_):
    data_ = read_csv('data/{}'.format(file_), header=None).dropna(axis=1)
    data_.columns = range(len(data_.columns))
    return data_.rename(columns={len(data_.columns)-1: 'label', 1: 'source', 2: 'target'})


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
        applymap(lambda x: [k for k in split('=| ', x) if k != ''])
    data_[range(3, len(data_.columns)-2)] = data_[range(3, len(data_.columns)-2)].\
        applymap(lambda list_: list_to_dict(list_))
    data_ = split_columns(data_)
    data_.columns = [sub('[,)(]', '', str(col_)) for col_ in data_.columns]

    return data_


def clean_text(text_):
    return sub('[^a-z0-9A-Z]+', ' ', text_).lower()


def tokenize_phrase(str_):
    return [word_ for word_ in str_.split() if word_ != '']


def get_vec_of_word(word_, model_):
    return model_.wv[word_]


def get_vectors(token_, phrase_, word_model=None, doc_model=None):
    list_of_vecs = map(get_vec_of_word, token_, len(token_)*[word_model])
    list_of_vecs.append(list(doc_model.infer_vector(phrase_.split())))

    return list_of_vecs


def calculate_mean_of_vecs(list_of_lists, dim_=None):
    sum_vec = np.zeros(dim_)
    for _, val_ in enumerate(list_of_lists):
        sum_vec = sum_vec + np.array(val_)

    return list(sum_vec/len(list_of_lists))


def get_vector_similarity(vec_1, vec_2):
    return cosine_similarity([map(float, vec_1)], [map(float, vec_2)])[0][0]


def calculate_similarity(data_, dim_=None):
    data_[['source', 'target']] = data_[['source', 'target']].applymap(lambda str_: clean_text(str_))
    data_[['source_token', 'target_token']] = data_[['source', 'target']].applymap(lambda str_: tokenize_phrase(str_))
    word2vec_model = Word2Vec(list(data_['source_token'].values)+list(data_['target_token'].values), min_count=1,
                              size=dim_, window=2)
    corpus_ = [TaggedDocument(doc_, [tag_]) for tag_, doc_ in enumerate(list(data_['source'].values) +
                                                                        list(data_['target'].values))]
    doc2vec_model = Doc2Vec(corpus_, vector_size=dim_, window=2, min_count=1, workers=4)

    data_['source_vec'] = data_.apply(lambda x: get_vectors(x['source_token'], x['source'], word_model=word2vec_model,
                                                            doc_model=doc2vec_model), axis=1)
    data_['target_vec'] = data_.apply(lambda x: get_vectors(x['target_token'], x['target'], word_model=word2vec_model,
                                                            doc_model=doc2vec_model), axis=1)
    data_[['source_vec', 'target_vec']] = data_[['source_vec', 'target_vec']].\
        applymap(lambda list_: calculate_mean_of_vecs(list_, dim_=dim_))
    data_['phrase_similarity'] = data_.apply(lambda x: get_vector_similarity(x['source_vec'], x['target_vec']), axis=1)
    data_ = data_.drop(columns=['source_token', 'target_token', 'source_vec', 'target_vec'])

    return data_


def oversample_data(data_, label_col=None):
    data_ = data_.drop(columns=['source', 'target'])
    label_dict = dict(Counter(data_[label_col]))
    max_label = [key_ for key_ in label_dict if label_dict[key_] == max(label_dict.values())][0]
    label_dict.pop(max_label)
    smt = SMOTE()
    sampled_data = data_[data_[label_col] == max_label]
    train_cols = list(data_.columns.difference({label_col}))
    for label_ in label_dict:
        _data = concat([data_[data_[label_col] == max_label], data_[data_[label_col] == label_]], ignore_index=True)
        data_x, data_y = smt.fit_sample(_data[train_cols], _data[label_col])
        _data = DataFrame(concat([DataFrame(data_x), Series(data_y)], axis=1).values, columns=train_cols+[label_col])
        sampled_data = sampled_data.append(_data[_data[label_col] != max_label], ignore_index=True)

    return sampled_data


def apply_pca(data_, percent_variance_to_capture=None):
    cols_ = list(set(data_.columns).difference(set(categorical_columns)))
    pca = PCA(n_components=len(cols_))
    scaled_data = scale(data_[cols_].values)
    pca.fit(scaled_data)
    cum_var = np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)
    num_comps = len(cum_var)-len([k for k in cum_var if k > percent_variance_to_capture])
    pca = PCA(n_components=num_comps)
    pca.fit(scaled_data)
    return concat([DataFrame(pca.fit_transform(scaled_data), columns=['PC'+'_'+str(k) for k in range(num_comps)]),
                   data_[categorical_columns]], axis=1)


def data_preprocessing(filename=None, dimension=None):
    data_ = load_data(filename)
    data_ = extract_values(data_)
    data_ = fix_duplicate_cols(data_)
    data_ = remove_cols(data_)
    data_ = replace_missing_values(data_)
    data_ = calculate_similarity(data_, dim_=dimension)
    data_ = concat([data_.drop(columns=['0', '14']), get_dummies(data_[['0', '14']])], axis=1)
    data_ = remove_cols(data_)
    data_ = oversample_data(data_, label_col='label')

    return data_
