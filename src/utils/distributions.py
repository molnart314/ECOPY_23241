import random
import math
import pyerf
import scipy

class UniformDistribution():
    def __init__(self, rand, a, b):
        self.rand = rand
        self.a = a
        self.b = b
    def pdf(self, x):
        self.x = x
        if self.a <= self.x <= self.b:
            self.x = 1 / (self.b - self.a)
        else:
            self.x = 0
        return self.x
    def cdf(self, x):
        self.x = x
        if self.a <= self.x <= self.b:
            self.x = (self.x - self.a) / (self.b - self.a)
        elif self.x < self.a:
                self.x = 0
        else:
                self.x = 1
        return self.x

    def ppf(self, p):
        self.p = p
        if 0 <= self.p <= 1:
            self.p = self.a + (self.b-self.a)*self.p
        return self.p

    def gen_random(self):
        return random.uniform(self.a, self.b)

    def mean(self):
        return 0.5*(self.a+self.b)

    def median(self):
        return 0.5*(self.a+self.b)

    def variance(self):
        return 1/12*(self.b-self.a)^2

    def skewness(self):
        return 0

    def ex_kurtosis(self):
        return -6/5

    def mvsk(self):
        return [0.5*(self.a+self.b), 1/12*(self.b-self.a)*(self.b-self.a), 0, -6/5]

class NormalDistribution:
    def __init__(self, rand, loc, scale):
        self.rand = rand
        self.loc = loc
        self.scale = scale


    def pdf(self, x):
        self.x = x
        sd = (self.scale) ** 0.5
        denominator = sd*(2*math.pi)**0.5
        exponent = (-0.5)*(((self.x-self.loc)/sd)**2)
        return 1/denominator*(math.e)**exponent

    def cdf(self, x):
        sd = (self.scale)**0.5
        denominator = sd*2**0.5
        self.x = x
        return 0.5*(1 + math.erf((self.x-self.loc)/denominator))

    def ppf(self, p):
        sd = (self.scale) ** 0.5
        denominator = sd * 2 ** 0.5
        self.p = p
        return pyerf.erfinv(2*p-1)*denominator+self.loc

    def gen_random(self):
        return random.normalvariate(self.loc, self.scale)

    def mean(self):
        return self.loc

    def median(self):
        return self.loc

    def variance(self):
        return self.scale

    def skewness(self):
        return 0

    def ex_kurtosis(self):
        return 0

    def mvsk(self):
        return [self.loc, self.scale, 0, 0]

class CauchyDistribution:
    def __init__(self, rand, loc, scale):
        self.rand = rand
        self.loc = loc
        self.scale = scale

    def pdf(self, x):
        self.x = x
        den_parentheses = (self.x-self.loc)/self.scale
        denominator = self.scale*math.pi*(1+den_parentheses**2)
        return 1/denominator

    def cdf(self, x):
        self.x = x
        parentheses = (self.x-self.loc)/self.scale

        return 1/math.pi*math.atan(parentheses) + 0.5

    def ppf(self, p):
        self.p = p
        parentheses = math.pi*(self.p - 0.5)
        return self.loc + self.scale*math.tan(parentheses)

    def gen_random(self):
        u = random.random()
        parentheses = math.pi * (u - 0.5)
        return self.loc + self.scale * math.tan(parentheses)

    def mean(self):
        raise Exception("Moments undefined")

    def median(self):
        raise Exception("Moments undefined")

    def skewness(self):
        raise Exception("Moments undefined")

    def ex_kurtosis(self):
        raise Exception("Moments undefined")

    def mvsk(self):
        return [self.mean(), self.median(), self.skewness(), self.ex_kurtosis()]

class LogisticDistribution:
    def __init__(self, rand, location, scale):
        self.rand = rand
        self.location = location
        self.scale = scale

    def pdf(self, x):
        if self.scale <= 0:
            raise ValueError("A szórás (scale) értéke pozitívnak kell lennie.")

        diff = x - self.location
        exponent = -diff / (2 * self.scale)
        sech_squared = 1 / (math.cosh(exponent) ** 2)
        pdf_value = sech_squared / (4 * self.scale)

        return pdf_value

    def cdf(self, x):
        if self.scale <= 0:
            raise ValueError("A szórás (scale) értéke pozitívnak kell lennie.")

        exponent = -(x - self.location) / self.scale
        cdf_value = 1 / (1 + math.exp(exponent))

        return cdf_value

    def ppf(self, p):
        if p < 0 or p > 1:
            raise ValueError("A valószínűségi érték (p) 0 és 1 között kell legyen.")

        if self.scale <= 0:
            raise ValueError("A szórás (scale) értéke pozitívnak kell lennie.")

        x = self.location - self.scale * math.log(1 / p - 1)
        return x

    def gen_rand(self):
        if self.scale <= 0:
            raise ValueError("A szórás (scale) értéke pozitívnak kell lennie.")

        u = self.rand.random()
        x = self.location - self.scale * math.log(1 / u - 1)

        return x

    def mean(self):
        return self.location

    def variance(self):
        return self.scale ** 2 * math.pi ** 2 / 3

    def skewness(self):
        return 0

    def ex_kurtosis(self):
        return 6 / 5

    def mvsk(self):
        variance = self.scale ** 2 * math.pi ** 2 / 3
        return [self.location, variance, 0, 6 / 5]

class ChiSquaredDistribution():
    def __init__(self, rand, dof):
        self.rand = rand
        self.dof = dof

    def pdf(self, x):
        if x < 0:
            return 0

        k = self.dof
        prefactor = 1 / (math.pow(2, k / 2) * scipy.special.gamma(k / 2))
        pdf_value = prefactor * math.pow(x, (k / 2) - 1) * math.exp(-x / 2)

        return pdf_value

    def cdf(self, x):
        if x < 0:
            return 0

        k = self.dof
        cdf_value = scipy.special.gammainc(k / 2, x / 2)

        return cdf_value

    def ppf(self, p):
        ppf_value = 2 * scipy.special.gammaincinv(self.dof / 2, p)
        return ppf_value

    def gen_rand(self):
        u = self.rand.random()
        x = 2 * scipy.special.gammaincinv(self.dof / 2, u)
        return x

    def mean(self):
        return self.dof

    def variance(self):
        return 2 * self.dof

    def skewness(self):
        return (8 / self.dof) ** 0.5

    def ex_kurtosis(self):
        return 12 / self.dof

    def mvsk(self):
        return [self.dof, 2 * self.dof, (8 / self.dof) ** 0.5, 12 / self.dof]
