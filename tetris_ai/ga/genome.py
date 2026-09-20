import random

import numpy as np

from .. import config

WEIGHT_INIT_RANGES = np.array([
    [0.0,   10.0],
    [0.1,   1.0],
    [-20.0, -10.0],
    [0.1,   2.0],
    [-5.0,  -1.0],
    [-5.0,  -1.0],
    [1.0,   10.0],
    [-10.0, -1.0],
    [-3.0,  -0.5],
    [-20.0, -5.0],
    [-10.0, -3.0],
    [0.0,   8.0],
], dtype=float)


def _random_weights():
    lo = WEIGHT_INIT_RANGES[:, 0]
    hi = WEIGHT_INIT_RANGES[:, 1]
    return (lo + (hi - lo) * np.random.random(len(lo))).astype(float)


class Genome:
    NUM_WEIGHTS = len(WEIGHT_INIT_RANGES)

    def __init__(self, weights=None):
        self.weights = (_random_weights() if weights is None
                        else np.array(list(weights), dtype=float))
        self.fitness = 0

    def mutate(self, mutation_rate=0.15, mutation_step=0.2, signflip_rate=0.02):
        for i in range(len(self.weights)):
            if random.random() < mutation_rate:
                w = self.weights[i]
                sigma = mutation_step * (abs(w) + mutation_step)
                self.weights[i] = w + np.random.normal(0.0, sigma)
                if random.random() < signflip_rate:
                    self.weights[i] = -self.weights[i]

    def crossover(self, partner):
        child_weights = [self.weights[i] if random.random() < 0.5 else partner.weights[i]
                         for i in range(len(self.weights))]
        return Genome(child_weights)

    def crossover_blx(self, partner, alpha=None):
        a = config.BLX_ALPHA if alpha is None else alpha
        w1 = np.asarray(self.weights, dtype=float)
        w2 = np.asarray(partner.weights, dtype=float)
        d = np.abs(w1 - w2)
        lo = np.minimum(w1, w2) - a * d
        hi = np.maximum(w1, w2) + a * d
        return Genome(list(lo + (hi - lo) * np.random.random(len(lo))))


def fmt_fit(x):
    x = float(x)
    if abs(x) >= 1e6:
        return f"{x / 1e6:.2f}M"
    if abs(x) >= 1e3:
        return f"{x / 1e3:.1f}k"
    return f"{x:.0f}"


def fmt_weights(w, ndigits=2):
    return "[" + ", ".join(f"{x:.{ndigits}f}" for x in w) + "]"
