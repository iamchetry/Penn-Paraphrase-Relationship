from phrase_relations.obtain_relation import *


d = load_data('ppdb.train.csv')
d = extract_values(d)
d = fix_duplicate_cols(d)
d = replace_missing_values(d)
d = remove_cols(d)
print d.columns
