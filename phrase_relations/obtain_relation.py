from phrase_relations.preprocess import *
from phrase_relations.modelling import *

import warnings
from gensim.models import Word2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

warnings.filterwarnings('ignore')


def calculate_similarity(data_, dim_=None):
    '''
    :param data_: data-frame
    :param dim_: embedding vector dimension
    :return: data-frame
    '''

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


def data_preprocessing(filename=None, dimension=None, columns_=None, percent_variance_to_capture=None, num_comps=None,
                       label_col=None):
    '''
    :param filename: csv file to import
    :param dimension: embedding vector dimension
    :param columns_: columns specified for test data
    :param percent_variance_to_capture: percent of total variance to capture
    :param num_comps: number of principal components specified for test data
    :param label_col: class label column name
    :return: (data-frame, list of columns, integer) or data-frame
    '''

    data_ = load_data(filename, label_col=label_col)
    data_ = extract_values(data_)
    data_ = fix_duplicate_cols(data_)
    if not columns_:
        columns_ = remove_cols(data_)
    data_ = replace_missing_values(data_[columns_])
    data_ = calculate_similarity(data_, dim_=dimension)
    data_ = concat([data_.drop(columns=['0', '14']), get_dummies(data_[['0', '14']])], axis=1)
    if 'train' in filename:
        data_ = oversample_data(data_, label_col=label_col)
    data_, num_comps = apply_pca(data_, percent_variance_to_capture=percent_variance_to_capture, num_comps=num_comps,
                                 label_col=label_col)
    if 'train' in filename:
        return data_, columns_, num_comps
    else:
        return data_


def obtain_train_test(train_filename=None, test_filename=None, dimension=None, percent_variance_to_capture=None,
                      label_col=None):
    '''
    :param train_filename: training csv file
    :param test_filename: testing csv file
    :param dimension: embedding vector dimension
    :param percent_variance_to_capture: percent of total variance to capture
    :param label_col: class label column name
    :return: two data-frames
    '''

    data_train, columns_, components_ = data_preprocessing(filename=train_filename, dimension=dimension,
                                                           percent_variance_to_capture=percent_variance_to_capture,
                                                           label_col=label_col)
    data_test = data_preprocessing(filename=test_filename, dimension=dimension, columns_=columns_, num_comps=components_,
                                   label_col=label_col)
    cols_in_common = take_common_cols(data_train, data_test)
    data_train, data_test = data_train[cols_in_common], data_test[cols_in_common]

    return data_train, data_test


def apply_model(train_data, model_name=None, label_col=None):
    '''
    :param train_data: training data-frame
    :param model_name: classification model name for dumping and loading
    :param label_col: class label column name
    :return: list of columns
    '''

    columns_ = [col_ for col_ in train_data.columns.difference({label_col})]
    param_learned = grid_search(train_data[columns_], train_data[label_col])
    dump_best_model(train_data[columns_], train_data[label_col], param_learned, model_name=model_name)

    return columns_


def phrase_relation_main(train_filename=None, test_filename=None, embedding_dimension=None, model_name=None,
                         percent_variance_to_capture=None, class_column=None):
    '''
    :param train_filename: training csv file
    :param test_filename: testing csv file
    :param embedding_dimension: embedding vector dimension
    :param model_name: classification model name for dumping and loading
    :param percent_variance_to_capture: percent of total variance to capture
    :param class_column: class label column name
    :return: dictionary containing score values
    '''

    data_train, data_test = obtain_train_test(train_filename=train_filename, test_filename=test_filename,
                                              dimension=embedding_dimension, label_col=class_column,
                                              percent_variance_to_capture=percent_variance_to_capture)
    columns_ = apply_model(data_train, model_name=model_name, label_col=class_column)
    test_predicted = prediction(data_test, model_name=model_name, columns_=columns_, label_col=class_column)
    return obtain_scores(actual_=test_predicted[class_column], predicted_=test_predicted[class_column+'_predicted'])
