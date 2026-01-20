from __future__ import annotations
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


# TODO:
#   - Fix issues with code where divisions by sine or cosine take place. This makes J2 computations invalid at certain
#     inclinations.
#   - Need to general lpmv array to generate it all at once.
class NonSphericalEarth(Perturbation):
    def __init__(self, order: int, degree: int, gmst: float):
        self.order = order
        self.degree = degree
        self.initial_gmst = gmst

        with importlib.resources.files("hohmannpy.resources").joinpath("egm84_c_coeffs.csv").open() as f:
            self.c_coeffs = np.loadtxt(f, delimiter=",")  # n-columns, m-rows, from [0, 180]

        with importlib.resources.files("hohmannpy.resources").joinpath("egm84_s_coeffs.csv").open() as f:
            self.s_coeffs = np.loadtxt(f, delimiter=",")  # n-columns, m-rows, from [0, 180]

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

    def compute_colat_and_long(self, time, position):
        earth_rot = 7.292115e-5  # Mean rotation rate of the Earth in radians.
        gmst = self.initial_gmst + earth_rot * time

        inertial_2_earth_dcm = dcms.euler_2_dcm(gmst, 3)
        position = inertial_2_earth_dcm @ position
        longitude = np.arctan2(position[1], position[0])
        colatitude = np.pi / 2 - np.arctan2(position[2], np.sqrt(position[0] ** 2 + position[1] ** 2))

        return colatitude, longitude


# TODO: Deal with the singularity at the poles.
class AtmosphericDrag(Perturbation):
    def __init__(
            self,
            ballistic_coeff: float,
            gmst: float,
            solar_activity: str = "moderate",
            solver_tol: float = 1e-8
    ):
        super().__init__()

        self.ballistic_coeff = ballistic_coeff
        self.initial_gmst = gmst
        self.solar_activity = solar_activity
        self.solver_tol = solver_tol

        match solar_activity:
            case "low":
                with importlib.resources.files("hohmannpy.resources").joinpath("cira_12_low_activity.csv").open() as f:
                    density_curve = np.loadtxt(f, delimiter=",")  # altitude (km), density (kg/m^3)
                    self.densities = sp.interpolate.make_interp_spline(
                        density_curve[:, 0].squeeze(),
                        density_curve[:, 1].squeeze(),
                        k=1
                    )
            case "moderate":
                with importlib.resources.files("hohmannpy.resources").joinpath("cira_12_moderate_activity.csv").open() as f:
                    density_curve = np.loadtxt(f, delimiter=",")  # altitude (km), density (kg/m^3)
                    self.densities = sp.interpolate.make_interp_spline(
                        density_curve[:, 0].squeeze(),
                        density_curve[:, 1].squeeze(),
                        k=1
                    )
            case "high":
                with importlib.resources.files("hohmannpy.resources").joinpath("cira_12_high_activity.csv").open() as f:
                    density_curve = np.loadtxt(f, delimiter=",")  # altitude (km), density (kg/m^3)
                    self.densities = sp.interpolate.make_interp_spline(
                        density_curve[:, 0].squeeze(),
                        density_curve[:, 1].squeeze(),
                        k=1
                    )
        self.exosphere_bound = density_curve[-1, 0]

    def evaluate(self, time: float, state: np.ndarray) -> tuple[float, float, float]:
        earth_rot = 7.292115e-5  # Mean rotation rate of the Earth in radians.
        gmst = self.initial_gmst + earth_rot * time

        inertial_2_earth_dcm = dcms.euler_2_dcm(gmst, 3)
        position = inertial_2_earth_dcm @ state[:3]
        altitude = self.compute_altitude(position)

        if altitude / 1000 > self.exosphere_bound:
            return 0, 0, 0

        density = self.densities(altitude / 1000)  # Need to convert m -> km

        velocity = state[3:] - np.cross(np.array([0, 0, earth_rot]), state[:3])

        acceleration = -0.5 * 1 / self.ballistic_coeff * density * np.linalg.norm(velocity) * velocity

        return acceleration[0], acceleration[1], acceleration[2]

    def compute_altitude(self, position: np.ndarray) -> float:
        earth_radius = 6378.1363e3
        earth_eccentricity = 0.081819221456

        x = np.arctan2(position[2], np.sqrt(position[0] ** 2 + position[1] ** 2))
        x_old = 100
        while abs(x - x_old) > self.solver_tol:
            x_old = x
            radius_of_curvature = earth_radius / np.sqrt((1 - earth_eccentricity ** 2 * np.sin(x) ** 2))
            x = np.arctan2(
                position[2] + radius_of_curvature * earth_eccentricity ** 2 * np.sin(x),
                np.sqrt(position[0] ** 2 + position[1] ** 2)
            )

        geodetic_latitude = x
        radius_of_curvature = earth_radius / np.sqrt((1 - earth_eccentricity ** 2 * np.sin(x) ** 2))
        altitude = np.sqrt(position[0] ** 2 + position[1] ** 2) / np.cos(geodetic_latitude) - radius_of_curvature

        return altitude


class ThirdBodyGravity(Perturbation):
    def __init__(self, grav_params: list[float], distances: list[float]):
        super().__init__()

        self.grav_params = grav_params
        self.distances = distances

    def evaluate(self, time: float, state: np.ndarray) -> tuple[float, float, float]:
        pass
