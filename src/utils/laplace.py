import random
import math


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
        if p <= 0 or p >= 1:
            raise ValueError("p must be between 0 and 1")

        if p < 0.5:
            return self.loc - self.scale * math.log(1 - 2 * p)
        else:
            return self.loc + self.scale * math.log(2 * p - 1)

    def gen_random(self):
        return self.generate_sample()

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