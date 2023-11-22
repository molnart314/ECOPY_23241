from math import copysign


def sign(input_value: float) -> float:
    return copysign(1, input_value)

    def gen_random(self):
        u = self.rand.random()  # Véletlen szám [0, 1) tartományban
        return self.ppf(u)