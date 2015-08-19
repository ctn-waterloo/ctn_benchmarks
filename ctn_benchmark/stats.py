import numpy as np

def find_offset(a, b):
    assert len(a) == len(b)
    corr = np.correlate(a, b, 'full')
    index = np.argmax(corr[len(a):])
    return index

def bootstrapci(data, func, n=3000, p=0.95):
    index=int(n*(1-p)/2)

    samples = np.random.choice(data, size=n)
    r = [func(s) for s in samples]
    r.sort()
    return r[index], r[-index]

