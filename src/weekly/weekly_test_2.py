#1

class LaplaceDistribution:
    def __init__(self, rand, loc, scale):
        self.rand = rand
        self.loc = loc
        self.scale = scale

    def generate_sample(self):
        u = self.rand.random()
        if u < 0.5:
            return self.loc + self.scale * (-(2 * u) ** 0.5)
        else:
            return self.loc - self.scale * (2 * (u - 0.5)) ** 0.5

    def pdf(self, x):
        abs_diff = abs(x - self.loc)
        return (1 / (2 * self.scale)) * math.exp(-abs_diff / self.scale)

    def cdf(self, x):
        if x < self.loc:
            return 0.5 * math.exp((x - self.loc) / self.scale)
        else:
            return 1 - 0.5 * math.exp(-(x - self.loc) / self.scale)

    def ppf(self, p):
        if p < 0 or p > 1:
            raise ValueError("p must be between 0 and 1")

        if p < 0.5:
            return self.loc + self.scale * math.log(2 * p)
        else:
            return self.loc - self.scale * math.log(2 - 2 * p)

    def gen_rand(self):
        u = self.rand.random()
        if u < 0.5:
            return self.loc + self.scale * (-(2 * u) ** 0.5)
        else:
            return self.loc - self.scale * (2 * (u - 0.5)) ** 0.5

    def mean(self):
        if self.scale == 0:
            raise Exception("Moment undefined")
        return self.loc

    def variance(self):
        if self.scale == 0:
            raise Exception("Moment undefined")
        return 2 * (self.scale ** 2)

    def skewness(self):
        if self.scale == 0:
            raise Exception("Moment undefined")
        return 0.0

    def ex_kurtosis(self):
        if self.scale == 0:
            raise Exception("Moment undefined")
        return 3.0

    def mvsk(self):
        if self.scale == 0:
            raise Exception("Moment undefined")

        mean = self.mean()
        variance = self.variance()
        skewness = self.skewness()
        kurtosis = self.ex_kurtosis()

        return [mean, variance, skewness, kurtosis]




#2
from random import choice
import random
import math

class ParetoDistribution:
    def __init__(self, rand, scale, shape):
        self.rand = rand
        self.scale = scale
        self.shape = shape

    def pdf(self, x):
        if x < self.scale:
            return 0.0
        else:
            return self.shape * (self.scale ** self.shape) / (x ** (self.shape + 1))

    def cdf(self, x):
        if x < self.scale:
            return 0.0
        else:
            return 1.0 - (self.scale / x) ** self.shape

    def ppf(self, p):
        if p < 0.0 or p > 1.0:
            raise ValueError("Probability p must be in the range [0, 1]")

        return self.scale / ((1.0 - p) ** (1.0 / self.shape))

    def gen_random(self):
        # Aszimmetrikus Laplace eloszlás generálása
        u1 = self.rand.random()  # Első véletlen szám [0, 1) tartományban
        u2 = self.rand.random()  # Második véletlen szám [0, 1) tartományban

        if u1 < 0.5:
            return self.scale * (-(2 * math.log(u2)) ** 0.5)
        else:
            return self.scale * ((2 * math.log(u2)) ** 0.5)

    def mean(self):
        if self.shape <= 1.0:
            raise Exception("Moment undefined")

        return (self.shape * self.scale) / (self.shape - 1.0)

    def variance(self):
        if self.shape <= 2.0:
            raise Exception("Moment undefined")

        return (self.scale ** 2) * (self.shape / ((self.shape - 1) ** 2) * (self.shape - 2))

    def skewness(self):
        if self.shape <= 3.0:
            raise Exception("Moment undefined")

        numerator = 2 * (1 + self.shape)
        denominator = (self.shape - 3) * (self.shape - 2) * (self.shape - 1)
        return numerator / denominator ** 0.5

    def ex_kurtosis(self):
        if self.shape <= 4.0:
            raise Exception("Moment undefined")

        numerator = 6 * (self.shape ** 2) * (self.shape + 1)
        denominator = (self.shape - 4) * (self.shape - 3) * (self.shape - 2) * (self.shape - 1)
        return numerator / denominator

    def mvsk(self):
        if self.scale == 0:
            raise Exception("Moment undefined")

        mean = self.mean()
        variance = self.variance()
        skewness = self.skewness()
        kurtosis = self.ex_kurtosis()

        return [mean, variance, skewness, kurtosis]



