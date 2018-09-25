import cPickle


def dump_as_pickle(filename, to_dump):
    '''
    :param filename: filename to be dumped as
    :param to_dump: object to be dumped
    :return: dump to folder 'model'
    '''

    with open("model/" + filename + ".pkl", 'wb') as fid:
        cPickle.dump(to_dump, fid)


def load_from_pickle(filename):
    '''
    :param filename: filename to load
    :return: object
    '''

    with open("model/" + filename + ".pkl", 'rb') as fid:
        return cPickle.load(fid)
