import pickle


def load_models(filename):
    with open(filename, 'rb') as fin:
        models = pickle.load(fin)
        return models


def save_models(models, filename):
    with open(filename, 'wb') as fout:
        pickle.dump(models, fout)
