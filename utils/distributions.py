"""@thematrixmaster
This file defines distributions used in this project
"""

import numpy as np
from abc import ABC, abstractmethod

np.random.seed(33)

class Distribution(ABC):
    @abstractmethod
    def pdf(self, x):
        pass
    
    @abstractmethod
    def sample(self, n):
        pass


class Dirichlet(Distribution):
    def __init__(self, alpha):
        self.alpha = alpha
        self.k = len(alpha)

    def pdf(self, x):
        return np.prod(np.power(x, self.alpha - 1))

    def sample(self, n):
        return np.random.dirichlet(self.alpha, n)


class Categorical(Distribution):
    def __init__(self, p):
        self.p = p
        self.k = len(p)

    def pdf(self, x):
        return np.prod(np.power(self.p, x))

    def sample(self, n):
        return np.random.multinomial(1, self.p, n).argmax(axis=1)