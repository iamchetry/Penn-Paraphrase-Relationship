from phrase_relations.constants import *
from phrase_relations.utilities import *

from pandas import *
from collections import Counter
from imblearn.over_sampling import SMOTE

from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def oversample_data(data_, label_col=None):
    '''
    :param data_: imbalanced data-frame
    :param label_col: class label column name
    :return: balanced oversampled data-frame
    '''

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


def apply_pca(data_, percent_variance_to_capture=None, num_comps=None, label_col=None):
    '''
    :param data_: data-frame
    :param percent_variance_to_capture: percent of total variance to capture
    :param num_comps: number of principal components for test data
    :param label_col: class label column name
    :return: concatenated data-frame
    '''

    cols_ = list(set([col_ for col_ in data_.columns if not ('0' in col_ or '14' in col_)]).
                 difference(set(categorical_columns)))
    if not num_comps:
        pca = PCA(n_components=len(cols_))
        scaled_data = scale(data_[cols_].values)
        pca.fit(scaled_data)
        cum_var = np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4) * 100)
        num_comps = len(cum_var) - len([k for k in cum_var if k > percent_variance_to_capture])

    pca = PCA(n_components=num_comps)
    scaled_data = scale(data_[cols_].values)
    pca.fit(scaled_data)
    return concat(
        [DataFrame(pca.fit_transform(scaled_data), columns=['PC' + '_' + str(k) for k in range(num_comps)]),
         data_[[col_ for col_ in data_.columns if ('0' in col_ or '14' in col_)] + ['ContainsX', label_col]]],
        axis=1), num_comps


def get_vec_of_word(word_, model_):
    '''
    :param word_: single word
    :param model_: word2vec model
    :return: word-vector
    '''

    return model_.wv[word_]


def get_vectors(token_, phrase_, word_model=None, doc_model=None):
    '''
    :param token_: list of word tokens
    :param phrase_: phrase
    :param word_model: word2vec model
    :param doc_model: doc2vec model
    :return: list of vectors
    '''

    list_of_vecs = map(get_vec_of_word, token_, len(token_)*[word_model])
    list_of_vecs.append(list(doc_model.infer_vector(phrase_.split())))

    return list_of_vecs


def get_vector_similarity(vec_1, vec_2):
    '''
    :param vec_1: first vector
    :param vec_2: second vector
    :return: cosine similarity value
    '''

    return cosine_similarity([map(float, vec_1)], [map(float, vec_2)])[0][0]


def grid_search(train_x, train_y):
    '''
    :param train_x: training data with independent variables
    :param train_y: training class labels
    :return: best parameters for classification model
    '''

    search_grid = GridSearchCV(estimator=RandomForestClassifier(), param_grid=param_dict, n_jobs=-1, cv=5)
    search_grid.fit(train_x, train_y)
    return search_grid.best_params_


def dump_best_model(train_x, train_y, best_params, model_name=None):
    '''

    :param train_x: training data with independent variables
    :param train_y: training class labels
    :param best_params: best parameters learned for classification model
    :param model_name: model name to dump
    :return: dump to folder 'model'
    '''

    model_ = RandomForestClassifier(n_jobs=-1).set_params(**best_params)
    model_.fit(train_x, train_y)
    dump_as_pickle(model_name, model_)


def prediction(test_data, model_name=None, columns_=None, label_col=None):
    '''
    :param test_data: testing data-frame
    :param model_name: classification model name for dumping and loading
    :param columns_: list of columns specified for testing data
    :param label_col: class label column name
    :return: concatenated data-frame
    '''

    model = load_from_pickle(model_name)
    return concat([test_data, DataFrame(model.predict(test_data[columns_]), columns=[label_col+'_predicted'])], axis=1)


def obtain_scores(actual_=None, predicted_=None):
    '''
    :param actual_: actual class labels
    :param predicted_: predicted class labels
    :return: dictionary with different score values
    '''

    return {'accuracy': accuracy_score(actual_, predicted_),
            'precision': precision_score(actual_, predicted_, average='macro'),
            'recall': recall_score(actual_, predicted_, average='macro'),
            'f1_score': f1_score(actual_, predicted_, average='macro')}