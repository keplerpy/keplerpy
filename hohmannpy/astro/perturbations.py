from abc import ABC, abstractmethod
import numpy as np
import scipy as sp
from ..dynamics import dcms
import importlib.resources


class Perturbation(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def evaluate(self, time: float, state: np.ndarray) -> tuple[float, float, float]:
        pass


class NonSphericalEarth(Perturbation):
    def __init__(self, order: int, degree: int, gmst: float):
        self.order = order
        self.degree = degree
        self.initial_gmst = gmst

        with importlib.resources.files("hohmannpy.resources").joinpath("egm84_c_coeffs.csv").open() as f:
            self.c_coeffs = np.loadtxt(f, delimiter=",")

        with importlib.resources.files("hohmannpy.resources").joinpath("egm84_s_coeffs.csv").open() as f:
            self.s_coeffs = np.loadtxt(f, delimiter=",")

        super().__init__()

    def evaluate(self, time: float, state: np.ndarray) -> tuple[float, float, float]:
        earth_radius = 6378137
        grav_param = 3.986004418e14

        radius = np.sqrt(state[0] ** 2 + state[1] ** 2 + state[2] ** 2)
        colatitude, longitude = self.compute_colat_and_long(time, state)

        radial_accel = 0
        longitudinal_accel = 0
        colatitudinal_accel = 0

        for n in range(1, self.order + 1):
            for m in range(0, self.degree + 1):
                radial_accel += (
                    -(n + 1) * grav_param * earth_radius ** n / radius ** (n + 2)
                        * sp.special.lpmv(m, n, np.cos(colatitude))
                        * (self.c_coeffs[n, m] * np.cos(m * longitude) + self.s_coeffs[n, m] * np.sin(m * longitude))
                )
                longitudinal_accel += (
                    1 / (radius * np.sin(colatitude))
                        * grav_param * earth_radius ** n / radius ** (n + 1)
                        * sp.special.lpmv(m, n, np.cos(colatitude))
                        * m
                        * (self.c_coeffs[n, m] * -np.sin(m * longitude) + self.s_coeffs[n, m] * np.cos(m * longitude))
                )
                if n == 0:
                    continue
                else:
                    colatitudinal_accel += (
                        1 / radius
                            * grav_param * earth_radius ** n / radius ** (n + 1)
                            * (n * np.cos(colatitude) * sp.special.lpmv(m, n, np.cos(colatitude))
                                - (n + m) * sp.special.lpmv(m, n - 1, np.cos(colatitude)))
                            / np.sin(colatitude)
                            * (self.c_coeffs[n, m] * np.cos(m * longitude)
                                + self.s_coeffs[n, m] * np.sin(m * longitude))
                    )

        curvilinear_accel = np.array([colatitudinal_accel, longitudinal_accel, radial_accel])
        curvilinear_2_rectilinear = dcms.euler_2_dcm(longitude, 3).T @ dcms.euler_2_dcm(colatitude, 2).T
        acceleration = curvilinear_2_rectilinear @ curvilinear_accel

        return acceleration[0], acceleration[1], acceleration[2]

    def compute_colat_and_long(self, time, state):
        earth_rot = 7.292115e-5  # Mean rotation rate of the Earth in radians.
        gmst = self.initial_gmst + earth_rot * time

        inertial_2_earth_dcm = dcms.euler_2_dcm(gmst, 3)
        position = inertial_2_earth_dcm @ state[:3]
        longitude = np.arctan2(position[1], position[0])
        colatitude = np.pi / 2 - np.arctan2(position[2], np.sqrt(position[0] ** 2 + position[1] ** 2))

        return colatitude, longitude
