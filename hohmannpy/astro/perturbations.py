from abc import ABC, abstractmethod
from numpy.typing import NDArray


class Perturbation(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def evaluate(self, time: float, state: NDArray[float]) -> tuple[float, float, float]:
        pass


class NonSphericalEarth(Perturbation):
    def __init__(self):
        super().__init__()

    def evaluate(self, time: float, state: NDArray[float]) -> tuple[float, float, float]:
        pass

    def compute_lat_and_long(self, time, state):
        pass
