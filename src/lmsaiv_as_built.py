"""
AsBuilt holds the dictionary of parameters derived during AIV testing which decribed the as-built
state of the LMS.  These include such things as plate scales, distortion transforms, etc.
"""
import pickle


class AsBuilt:

    pickle_path = '../output/asbuilt/asbuilt.pkl'
    slice_bounds = None

    def __init__(self):
        return

    @staticmethod
    def write():
        file = open(AsBuilt.pickle_path, 'wb')
        pickle.dump(AsBuilt.slice_bounds, file)
        file.close()
        return

    @staticmethod
    def read():
        file = open(AsBuilt.pickle_path, 'wb')
        AsBuilt.slice_bounds = pickle.load(file)
        file.close()
        return
