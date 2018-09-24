from phrase_relations.constants import *
from phrase_relations.utilities import *

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def grid_search(train_x, train_y):
    search_grid = GridSearchCV(estimator=RandomForestClassifier(), param_grid=param_dict, n_jobs=-1, cv=5)
    search_grid.fit(train_x, train_y)
    return search_grid.best_params_


def dump_best_model(train_x, train_y, best_params, model_name=None):
    model_ = RandomForestClassifier(n_jobs=-1).set_params(**best_params)
    model_.fit(train_x, train_y)
    dump_as_pickle(model_name, model_)


def obtain_scores(actual_=None, predicted_=None):
    return {'accuracy': accuracy_score(actual_, predicted_),
            'precision': precision_score(actual_, predicted_, average='macro'),
            'recall': recall_score(actual_, predicted_, average='macro'),
            'f1_score': f1_score(actual_, predicted_, average='macro')}