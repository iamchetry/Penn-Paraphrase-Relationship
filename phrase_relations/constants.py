categorical_columns = ['0', 'source', 'target', '14', 'ContainsX', 'label']

param_dict = {'n_estimators': [10, 20, 30],
              'criterion': ['gini', 'entropy'],
              'max_features': ['auto', 'sqrt', 'log2'],
              'min_samples_split': [2, 4, 6],
              'min_samples_leaf': [1, 2],
              'bootstrap': [True, False]}
