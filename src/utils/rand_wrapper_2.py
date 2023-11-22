import pyerf
import typing
import random
import math


random.random()

random.randint(1,100)

random.seed(42)
random.randint(1,100)

def random_from_list(input_list):
    return random.choice(input_list)

def random_sublist_from_list(input_list, number_of_elements):
    return random.sample(input_list, number_of_elements)

def random_from_string(input_string):
    return random.choice(input_string)

def hundred_small_random():
    x = [random.random() for i in range(100)]
    return x

def hundred_large_random():
    x = [random.randint(10,1000) for i in range(100)]
    return x

def five_random_number_div_three():
    by_three=[]
    while len(by_three) <5:
        elem=random.randrange(9, 999, 3)
        by_three.append(elem)
    return by_three
#def five_random_number_div_three():
  #  by_three=[]
  #  while len(by_three) <5:
  #      elem=random.randint(9,1000)
  #      if elem%3==0:
  #          by_three.append(elem)
 #   return by_three

def random_reorder(input_list):
    return random.sample(input_list, len(input_list))

def uniform_one_to_five():
    return random.uniform(1,6)

####################################DISTRIBUTIONS#############################xx

class FirstClass:
    pass

class SecondClass:
    def __init__ (self, rand):
        self.random = rand
class UniformDistribution:
    def __init__(self, rand, a, b):
        self.rand = rand
        self.a = a
        self.b = b

    def pdf(self, x):
        if x >= self.a and x <= self.b:
            return 1.0 / (self.b - self.a)
        else:
            return 0.0
    def cdf(self, x):
        if x < self.a:
            return 0.0
        elif x >= self.a and x <= self.b:
            return (x - self.a) / (self.b - self.a)
        else:
            return 1.0

    def ppf(self, p):
        if 0 <= p <= 1:
            raise ValueError("p must be between 0 and 1")
        return self.a + p * (self.b - self.a)

    def gen_random(self):
        return self.rand.uniform(self.a, self.b)

    def mean(self):
        if self.a == self.b:
          raise Exception("Moment undefined")
        return (self.a + self.b) / 2.0

    def median(self):
        return (self.b + self.a) / 2

    def variance(self):
        if self.a == self.b:
            raise Exception("Moment undefined")
        return ((self.b - self.a) ** 2) / 12
    #return (self.b - self.a) / 12

    def skewness(self):
        if self.a == self.b:
            raise Exception("Moment undefined")
        return 0.0
    def ex_kurtosis(self):
        if self.a == self.b:
            raise Exception("Moment undefined")
        return 1.8-3

    def mvsk(self):
        if self.a == self.b:
            raise Exception("Moments undefined")
        mean = self.mean()
        variance = self.variance()
        skewness = self.skewness()
        ex_kurtosis = self.ex_kurtosis()
        return [mean, variance, skewness, ex_kurtosis]


import math
import random
from pyerf import pyerf
class NormalDistribution:
    def __init__(self, rand, loc, scale):
        self.rand = rand
        self.loc = loc
        self.scale = scale

    def pdf(self, x):
        return (1 / (self.scale**0.5 * ((2 * math.pi) ** 0.5))) * (math.e ** (-0.5 * (((x - self.loc) / self.scale**0.5) ** 2)))

    def cdf(self, x):
        return 0.5 * (1 + math.erf((x - self.loc) / (self.scale**0.5 * (2 ** 0.5))))

    def ppf(self, p):
        if p < 0 or p > 1:
            raise ValueError("p must be between 0 and 1")
        return self.loc + self.scale**0.5 * (2 ** 0.5) * pyerf.erfinv(2 * p - 1)

    def gen_rand(self):
        return random.normalvariate(self.loc,self.scale)

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
        return [self.mean(), self.variance(), self.skewness(), self.ex_kurtosis()]


class CauchyDistribution:
    def __init__(self, rand, x0, gamma):
        self.rand = rand
        self.location = x0
        self.scale = gamma

    def pdf(self, x):
        if self.scale <= 0:
            return 0.0
        else:
            return 1.0 / (math.pi * self.scale * (1.0 + ((x - self.location) / self.scale) ** 2))

    def cdf(self, x):
        if self.scale <= 0:
            return 0.0
        else:
            return 0.5 + (1.0 / math.pi) * math.atan((x - self.location) / self.scale)

    def ppf(self, p):
        if p < 0 or p > 1:
            raise ValueError("Az 'p' értéke 0 és 1 között kell legyen.")
        if p == 0.5:
            return self.location
        return self.location + self.scale * math.tan(math.pi * (p - 0.5))


    def gen_rand(self):
        return self.loc + self.scale * math.tan(math.pi * (self.rand.random() - 0.5))

    def mean(self):
        raise Exception("Moment undefined")

    def median(self):
        return self.location

    def variance(self):
        raise Exception("Moment undefined")

    def skewness(self):
        raise Exception("Moment undefined")

    def ex_kurtosis(self):
        raise Exception("Moment undefined")

    def mvsk(self):
        raise Exception("Moments undefined")



