import os

import numpy as np

class Data(object):
    def __init__(self, path, filenames=None, label=None):
        self.path = path
        if label is None:
            label = path
        self.label = label
        self.data = []
        if os.path.exists(path):
            self.filenames = os.listdir(path)
            if filenames is not None:
                self.filenames = set.intersection(set(self.filenames),
                                                  set(filenames))
            for fn in self.filenames:
                if fn.endswith('.txt'):
                    with open(os.path.join(path, fn)) as f:
                        text = f.read()
                    d = dict()
                    try:
                        exec(text, d)
                    except:
                        continue
                    del d['__builtins__']
                    self.data.append(d)

    def find_outliers(self, measures, cutoff=3.0):
        keep = np.ones(len(self.data))
        for m in measures:
            data = self.get(m)
            keep[np.abs(data - np.mean(data)) > cutoff * np.std(data)] = 0
        return np.arange(len(self.data))[keep == 0]

    def remove_outliers(self, measures, cutoff=3.0):
        keep = np.ones(len(self.data))
        for m in measures:
            data = self.get(m)
            keep[np.abs(data - np.mean(data)) > cutoff * np.std(data)] = 0
        all_data = []
        for i, d in enumerate(self.data):
            if keep[i] > 0:
                all_data.append(d)
        print('Removed %d outliers' % (len(self.data) - len(all_data)))
        self.data = all_data


    def get(self, key):
        return np.array([d[key] for d in self.data])
