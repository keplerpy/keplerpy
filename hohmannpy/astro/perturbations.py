from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np
import scipy as sp
from ..dynamics import dcms
import importlib.resources


class Perturbation(ABC):
    """
    Base class for implementing perturbations from Keplerian two-body orbital mechanics. These are designed to be used
    in conjunction with a non-Keplerian propagator such as :class:`~hohmannpy.astro.CowellPropagator` or
    :class:`~hohmannpy.astro.EnckePropagator`.

    Child classes must implement :meth:`evaluate()`. This is called on each timestep of propagation by
    :class:`~hohmannpy.astro.Propagator` . :meth:`~hohmannpy.astro.Propagator.propagate()` to return thw perturbing
    acceleration for a given orbital state.
    """

    def __init__(self):
        pass

    @abstractmethod
    def evaluate(self, time: float, state: np.ndarray) -> tuple[float, float, float]:
        """
        Takes in the current time and planet-centered inertial state (the position and velocity) and returns the
        perturbing acceleration.

        Parameters
        ----------
        time : float
            Current time in seconds since propagation began.
        state : np.ndarray
            Current translational state in planet-centered inertial coordinates given as (position, velocity).
        """

        pass


# TODO: Fix issues with code where divisions by sine takes place.
# TODO: This function not working.
class NonSphericalEarth(Perturbation):
    """
    Perturbation caused by the deviations of the Earth's math distribution from a point-mass. It is assumed that the
    gravitational potential field of the non-spherical Earth is given by the solution to a geopotential
    partial-differential equation and in addition that this field is conservative such that the perturbing force is it's
    gradient.

    The geopotential equation is a partial differential equation of three independent variables, spherical coordinates.
    Since this equation is separable it's solution is an infinite series whose coefficients, known as Stokes
    coefficients (C and S), have been determined analytically. This implementation uses the 1984 Earth Gravitational
    Model (EGM84) which includes harmonics up to order and degree 180. See the Notes section for how to upload your own
    survey data.

    In addition, to determine the geopotential at a given point the colatitude and longitude of the satellite must be
    known. This requires knowledge of the current GMST (angle between the Greenwich meridian and the Vernal equinox) of
    the Earth. For simplicity, the GMST is located accurately (including precession of the Vernal equinox) at the start
    of the simulation. However, for the length of propagation it is said to simply rotate at the Earth's mean rotation
    rate, ignoring precession effects.

    Parameters
    ----------
    degree : int
        Maximum degree of harmonics to include.
    gmst : float
        Current angle of the Greenwich meridian in :math:`rad`.
    zonal : bool
        Disable sectoral and tesseral harmonics to only look at zonal ones (such as J2). Does this by capping the
        maximum order summed to when computing the acceleration terms to 0.

    Attributes
    ----------
    degree : int
        Maximum degree of harmonics to include.
    initial_gmst : float
        Initial angle of the Greenwich meridian in :math:`rad` when propagation began.
    zonal : bool
        Disable sectoral and tesseral harmonics to only look at zonal ones (such as J2). Does this by capping the
        maximum order summed to when computing the acceleration terms to 0.
    c_coeffs : np.ndarray
        Cosine-like Stokes coefficients (unnormalized) from EGM84.
    s_coeffs : np.ndarray
        Sine-like Stokes coefficients (unnormalized) from EGM84.
    """

    def __init__(self, degree: int, gmst: float, zonal: bool = False):
        super().__init__()

        self.degree = degree
        self.zonal = zonal
        self.initial_gmst = gmst

        with importlib.resources.files("hohmannpy.resources").joinpath("egm84_c_coeffs.csv").open() as f:
            self.c_coeffs = np.loadtxt(f, delimiter=",")  # n rows, m columns, from [0, 180]

        with importlib.resources.files("hohmannpy.resources").joinpath("egm84_s_coeffs.csv").open() as f:
            self.s_coeffs = np.loadtxt(f, delimiter=",")  # n rows, m columns, from [0, 180]

    def evaluate(self, time: float, state: np.ndarray) -> tuple[float, float, float]:
        """
        Computes the perturbing acceleration using a geopotential model of the Earth's gravitational field.

        First the colatitude and longitude are found from the current time and state using
        :meth:`compute_colat_and_long`. The, the perturbing accelerations are computed in Earth-centered Earth-fixed
        (ECEF) curvilinear/spherical coordinates. Finally, this is transformed back to rectilinear adn then
        Earth-centered inertial (ECI) coordinates using DCMs generated via :func:`~hohmannpy.dynamics.dcms.euler_2_dcm()`.

        Parameters
        ----------
        time : float
            Current time in seconds since propagation began.
        state : np.ndarray
            Current translational state in PCI coordinates given as (position, velocity).
        """

        earth_radius = 6378137
        grav_param = 3.986004418e14
        earth_rot = 7.292115e-5  # Mean rotation rate of the Earth in rad/s.

        # Compute the colatitude and longitude.
        radius = np.sqrt(state[0] ** 2 + state[1] ** 2 + state[2] ** 2)
        colatitude, longitude = self.compute_colat_and_long(time, state[:3])

        # Compute the needed Legendre functions and their derivatives.
        legendre_funcs = sp.special.assoc_legendre_p_all(self.degree, self.degree, np.cos(colatitude), diff_n=1)

        # Compute the acceleration in curvilinear coordinates. Since the potential field and hence acceleration is an
        # infinite series in order and degree, iterate through both of these for all three components of the
        # acceleration.
        radial_accel = 0
        longitudinal_accel = 0
        colatitudinal_accel = 0

        for n in range(2, self.degree + 1):  # Degree 0 is point-mass, degree 1 is always 0, so skip.
            if self.zonal:  # Custom order limiter to allow for only inspecting zonal harmonics.
                m_lim = 1
            else:
                m_lim = n + 1
            for m in range(0, m_lim):
                radial_accel += (
                    -(n + 1) * grav_param / radius ** 2 * (earth_radius / radius) ** n
                        * legendre_funcs[0, n, m]
                        * (self.c_coeffs[n, m] * np.cos(m * longitude) + self.s_coeffs[n, m] * np.sin(m * longitude))
                )
                longitudinal_accel += (
                    1 / (radius ** 2 * np.sin(colatitude))
                        * grav_param * (earth_radius / radius) ** n
                        *  legendre_funcs[0, n, m]
                        * m
                        * (self.c_coeffs[n, m] * -np.sin(m * longitude) + self.s_coeffs[n, m] * np.cos(m * longitude))
                )
                colatitudinal_accel += (
                    -1 / radius ** 2
                        * grav_param * (earth_radius / radius) ** n
                        * legendre_funcs[1, n, m] * np.sin(colatitude)
                        * (self.c_coeffs[n, m] * np.cos(m * longitude) + self.s_coeffs[n, m] * np.sin(m * longitude))
                )

        # Use a DCM to convert back to rectilinear coordinates.
        curvilinear_accel = np.array([colatitudinal_accel, longitudinal_accel, radial_accel])
        curvilinear_2_rectilinear = dcms.euler_2_dcm(longitude, 3).T @ dcms.euler_2_dcm(colatitude, 2).T
        acceleration = curvilinear_2_rectilinear @ curvilinear_accel

        # Acceleration is still fixed to the Earth, need to now convert to an inertial basis.
        gmst = self.initial_gmst + earth_rot * time
        earth_2_inertial_dcm = dcms.euler_2_dcm(gmst, 3).T
        acceleration = earth_2_inertial_dcm @ acceleration

        return acceleration[0], acceleration[1], acceleration[2]

    def compute_colat_and_long(self, time, position):
        """
        Computes the colatitude and longitude of the satellite wrt. the Greenwich meridian from the PCI position and
        time.

        Parameters
        ----------
        time : float
            Current time in seconds since propagation began.
        position : np.ndarray
            Current PCI position.
        """

        earth_rot = 7.292115e-5  # Mean rotation rate of the Earth in rad/s.

        # Update GMST using simplified precession-free rotation of the Earth.
        gmst = self.initial_gmst + earth_rot * time

        # Transform position to the Earth-centered-Earth-fixed frame.
        inertial_2_earth_dcm = dcms.euler_2_dcm(gmst, 3)
        position = inertial_2_earth_dcm @ position

        # Compute longitude and colatitude.
        longitude = np.arctan2(position[1], position[0])
        colatitude = np.pi / 2 - np.arctan2(position[2], np.sqrt(position[0] ** 2 + position[1] ** 2))

        return colatitude, longitude


# TODO: Documentation for this class.
class J2(Perturbation):
    def __init__(self, gmst):
        self.initial_gmst = gmst
        super().__init__()

    def evaluate(self, time: float, state: np.ndarray) -> tuple[float, float, float]:
        earth_radius = 6378.1363e3
        earth_rot = 7.292115e-5  # Mean rotation rate of the Earth in radians.
        grav_param = 3.986004418e14
        j2 = 1.08e-3

        radius = np.linalg.norm(state[:3])

        gmst = self.initial_gmst + earth_rot * time
        inertial_2_earth_dcm = dcms.euler_2_dcm(gmst, 3)
        position = inertial_2_earth_dcm @ state[:3]

        acceleration = -3 * j2 * grav_param * earth_radius ** 2 / (2 * radius ** 5) * np.array([
            position[0] * (1 - 5 * position[2] ** 2 / radius ** 2),
            position[1] * (1 - 5 * position[2] ** 2 / radius ** 2),
            position[2] * (3 - 5 * position[2] ** 2 / radius ** 2)
        ]
        )

        acceleration = inertial_2_earth_dcm.T @ acceleration

        return acceleration[0], acceleration[1], acceleration[2]


# TODO: Deal with the singularity at the poles.
class AtmosphericDrag(Perturbation):
    """
    Perturbation caused by drag due to Earth's atmosphere.

    The Earth's atmosphere, especially the thermo- and exosphere, have highly variable properties due to fluctuations in
    the Earth's magnetic field, solar activity, and the position of the Earth along its orbit. The density is needed to
    compute the drag and is found via linear interpolation of the 2012 Committee on Space Research (COSPAR)
    International Reference Atmosphere (CIRA-12) model. Three different tables are provided, each representing a varying
    level of solar and geomagnetic activity.

    Two additional model simplifications are made. The use of a constant ballistic coefficient also simplifies the model
    by removing drag-attitude dependence. Also, computing geodetic latitude (which is needed to get an accurate
    altitude measurement) involves knowing the GMST (the angle between the Greenwich meridian and Vernal equinox) of the
    Earth. For simplicity, the GMST is located accurately (including precession of the Vernal equinox) at the start of
    the simulation. However, for the length of propagation it is said to simply rotate at the Earth's mean rotation
    rate, ignoring precession effects.

    Parameters
    ----------
    ballistic_coeff : float
        Drag times reference area of the satellite normalized by the mass.
    gmst : float
        Current angle of the Greenwich meridian in radians.
    solar_activity : str
        Which CIRA-12 reference atmosphere model to use for the density. Can select between "low", "medium", and "high".
        See the CIRA-12 offical report [1]_ for more details on how to select between these.
    solver_tol : float
        Tolerance to use when solving for the geodetic latitude via fixed-point iteration.

    Attributes
    ----------
    ballistic_coeff : float
        Drag times reference area of the satellite normalized by the mass.
    initial_gmst : float
        Initial angle of the Greenwich meridian in :math:`rad` when propagation began.
    solver_tol : float
        Tolerance to use when solving for the geodetic latitude via fixed-point iteration.
    densities : scipy.BSpline
        Piece-wise linear spline generated from a density curve where the independent variable is altitudes in
        :math:`km` and the dependent variable is densities in :math:`kg/m^3`.
    exosphere_bound : float
        Upper limit of the exosphere in :math:`km` above which the density is assumed to be zero and hence there is no
        drag.

    Notes
    -----
    The altitude above an ellipsoid Earth is found using Algorith 12 in Vallado [2]_.

    .. [1] COSPAR, COSPAR International Reference Atmosphere – CIRA-2012, Version: 1.0, spacewx.com, 2012.
    .. [2] Vallado, D. A., Fundamentals of Astrodynamics and Applications, 3rd ed., Microcosm Press/Springer, 2007.
    """

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
        """
        Computes the perturbing acceleration using a model drag caused by the Earth's atmosphere.

        The geodetic altitude is first found using :meth:`compute_altitude()` assuming an ellipsoid Earth and then the
        density is found via interpolation of atmospheric data. Using this the velocity wrt. the relative wind and then
        acceleration due to drag are found.

        Parameters
        ----------
        time : float
            Current time in seconds since propagation began.
        state : np.ndarray
            Current translational state in PCI coordinates given as (position, velocity).
        """

        earth_rot = 7.292115e-5  # Mean rotation rate of the Earth in radians.

        # Update GMST using simplified precession-free rotation of the Earth.
        gmst = self.initial_gmst + earth_rot * time

        # Transform position to the Earth-centered-Earth-fixed frame and then compute the altitude.
        inertial_2_earth_dcm = dcms.euler_2_dcm(gmst, 3)
        position = inertial_2_earth_dcm @ state[:3]
        altitude = self.compute_altitude(position)

        if altitude / 1000 > self.exosphere_bound:  # Effectively no atmosphere above this altitude.
            return 0, 0, 0

        # Compute density.
        density = self.densities(altitude / 1000)  # Need to convert m -> km

        # Compute velocity using a simplified approximation where the atmosphere is assumed fixed to the Earth rotating
        # at its mean rotation rate.
        velocity = state[3:] - np.cross(np.array([0, 0, earth_rot]), state[:3])

        # Compute perturbing acceleration.
        acceleration = -0.5 * 1 / self.ballistic_coeff * density * np.linalg.norm(velocity) * velocity

        return acceleration[0], acceleration[1], acceleration[2]

    def compute_altitude(self, position: np.ndarray) -> float:
        """
        Compute the altitude above the surface of an ellipsoid Earth.

        The geodetic latitude can be found as a function of the satellite's current position, however this function is
        transcendental in latitude and hence must be solved numerically. Fixed-point iteration is used. Once the
        latitude is known the ellipsoidal altitude may be computed.

        Notes
        -----
        This is less accurate than using the true altitude based on a series expansion (similar to the non-spherical
        geopotential gravity equation) but the accuracy loss is small.
        """

        earth_radius = 6378.1363e3
        earth_eccentricity = 0.081819221456

        # Compute initial guess for the latitude.
        x = np.arctan2(position[2], np.sqrt(position[0] ** 2 + position[1] ** 2))
        x_old = 100  # Dummy value to ensure error is initially above tolerance.

        # Perform fixed-point iteration.
        while abs(x - x_old) > self.solver_tol:
            x_old = x
            radius_of_curvature = earth_radius / np.sqrt((1 - earth_eccentricity ** 2 * np.sin(x) ** 2))
            x = np.arctan2(
                position[2] + radius_of_curvature * earth_eccentricity ** 2 * np.sin(x),
                np.sqrt(position[0] ** 2 + position[1] ** 2)
            )

        # Using the latitude compute the ellipsoidal altitude.
        geodetic_latitude = x
        radius_of_curvature = earth_radius / np.sqrt((1 - earth_eccentricity ** 2 * np.sin(x) ** 2))
        altitude = np.sqrt(position[0] ** 2 + position[1] ** 2) / np.cos(geodetic_latitude) - radius_of_curvature

        return altitude


# TODO: This function.
class ThirdBodyGravity(Perturbation):
    def __init__(self, grav_params: list[float], distances: list[float]):
        super().__init__()

        self.grav_params = grav_params
        self.distances = distances

    def evaluate(self, time: float, state: np.ndarray) -> tuple[float, float, float]:
        pass
