import math


class Parameter:
    def __call__(self, *args, **kwargs):
        pass


class ConstantParameter(Parameter):
    def __init__(self, value) -> None:
        self.value = value

    def __call__(self, *args, **kwargs) -> float:
        return self.value


class ExpDecayParameter(Parameter):
    def __init__(self, initial_value: float = 1, half_life: int = 100, final_value: float = 0) -> None:
        self.half_life = half_life
        self.initial_value = initial_value
        self.final_value = final_value

    def __call__(self, episode) -> float:
        return (self.initial_value - self.final_value) * pow(math.e, - episode * math.log(2) / self.half_life) + self.final_value
