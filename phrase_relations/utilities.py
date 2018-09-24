import cPickle


def dump_as_pickle(filename, to_dump):
    with open("model/" + filename + ".pkl", 'wb') as fid:
        cPickle.dump(to_dump, fid)


def load_from_pickle(filename):
    with open("model/" + filename + ".pkl", 'rb') as fid:
        return cPickle.load(fid)
