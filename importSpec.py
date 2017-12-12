#Small utility function to import pickled spectogram matrix
import pickle

def importSpec(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
        return data