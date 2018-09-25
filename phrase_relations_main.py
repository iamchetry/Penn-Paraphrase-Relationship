from phrase_relations.obtain_relation import *


print phrase_relation_main(train_filename='ppdb.train.csv', test_filename='ppdb.test.csv', embedding_dimension=300,
                           model_name='model_rf', percent_variance_to_capture=95, class_column='label')
