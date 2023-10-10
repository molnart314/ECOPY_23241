
#1 LogisticDistribution

import random
import math
class LogisticDistribution:
    def __init__(self, rand, location, scale):
        self.rand = rand
        self.location = location  # A location paramétert attribútumként tároljuk
        self.scale = scale  # A scale paramétert attribútumként tároljuk


    def pdf(self, x):
        if self.scale <= 0:
            raise ValueError("Scale (szórás) értéke pozitívnak kell lennie.")

        exponent = math.exp(-(x - self.location) / self.scale)
        denominator = self.scale * (1 + exponent) ** 2
        return exponent / denominator

    def cdf(self, x):
        if self.scale <= 0:
            raise ValueError("Scale (szórás) értéke pozitívnak kell lennie.")

        z = (x - self.location) / self.scale
        return 1 / (1 + math.exp(-z))

    def ppf(self, p):
        if self.scale <= 0:
            raise ValueError("Scale (szórás) értéke pozitívnak kell lennie.")
        if p < 0 or p > 1:
            raise ValueError("Az 'p' értéke 0 és 1 között kell legyen.")

        z = math.log(p / (1 - p))
        x = self.location + self.scale * z
        return x

    def gen_rand(self):
        if self.scale <= 0:
            raise ValueError("Scale (szórás) értéke pozitívnak kell lennie.")

        u = self.rand.random()  # Véletlen szám [0, 1) tartományban
        z = math.log(u / (1 - u))
        x = self.location + self.scale * z
        return x

    def mean(self):
        if self.scale <= 0:
            raise Exception("Moment undefined")
        return self.location

    def variance(self):
        if self.scale <= 0:
            raise Exception("Moment undefined")
        return (math.pi ** 2) * (self.scale ** 2) / 3

    def skewness(self):
        if self.scale <= 0:
            raise Exception("Moment undefined")
        return 0.0  # A logisztikus eloszlás ferdesége mindig 0, nem szükséges számítani

    def ex_kurtosis(self):
        return 1.2

    def mvsk(self):
        mean = self.mean()
        variance = self.variance()
        skewness = self.skewness()
        excess_kurtosis = self.ex_kurtosis()
        return [mean, variance, skewness, excess_kurtosis]

#2 ChiSquaredDistribution

import random
import typing
import math
import scipy.special as sp


class ChiSquaredDistribution:
    def __init__(self, rand, dof):
        self.rand = rand
        self.dof = dof

    def pdf(self, x):
        if x < 0:
            return 0.0  # Sűrűségfüggvény nulla negatív értékeken

        # A Chi-négyzet eloszlás sűrűségfüggvényének számítása
        numerator = x ** ((self.dof / 2) - 1) * math.exp(-x / 2)
        denominator = (2 ** (self.dof / 2)) * sp.gamma(self.dof / 2)
        return numerator / denominator

    def cdf(self, x):
        if x < 0:
            return 0.0

        cdf_value = sp.gammainc(self.dof / 2,
                                x / 2)
        return cdf_value

    def ppf(self, p):
        if p < 0 or p > 1:
            raise ValueError("Az 'p' értéke 0 és 1 között kell legyen.")

        ppf_value = 2 * sp.gammaincinv(self.dof / 2,
                                       p)
        return ppf_value

    def gen_rand(self):
        # Generate a random number using a method of your choice
        # For example, you can use the inverse transform sampling method
        u = self.rand.random()
        return self.ppf(u)

    def mean(self):
        if self.dof <= 0:
            raise Exception("Moment undefined")
        return self.dof

    def variance(self):
        if self.dof <= 0:
            raise Exception("Moment undefined")
        return 2 * self.dof  # A variancia kiszámítása

    def skewness(self):
        if self.dof <= 0:
            raise Exception("Moment undefined")
        return math.sqrt(8.0 / self.dof)  # Ferdeség számítása

    def ex_kurtosis(self):
        if self.dof <= 0:
            raise Exception("Moment undefined")
        return 12.0 / self.dof  # Többlet csúcsosság számítása

    def mvsk(self):
        if self.dof <= 0:
            raise Exception("Moment undefined")

        first_moment = self.dof
        variance = 2 * self.dof
        skewness = math.sqrt(8.0 / self.dof)
        ex_kurtosis = 12.0 / self.dof

        return [first_moment, variance, skewness, ex_kurtosis]
