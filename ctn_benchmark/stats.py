import numpy as np

def find_offset(a, b):
    assert len(a) == len(b)
    corr = np.correlate(a, b, 'full')
    index = np.argmax(corr[len(a):])
    return index

