import numpy as np

class Signal(object):
    def __init__(self, D, L, dt, max_freq, seed=None):
        rng = np.random.RandomState(seed=seed)
        steps = int(max_freq * L)
        self.w = 2 * np.pi * np.arange(steps) / L
        self.A = rng.randn(D, steps) + 1.0j * rng.randn(D, steps)

        power = np.sqrt(np.sum(self.A * self.A.conj()))
        self.A /= power

    def value(self, t):
        s = np.sin(self.w * t) * self.A
        return np.sum(s, axis=1).real
    def dvalue(self, t):
        s = np.cos(self.w * t) * self.w * self.A
        return np.sum(s, axis=1).real
