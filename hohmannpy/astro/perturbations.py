from __future__ import annotations
from abc import ABC, abstractmethod
import importlib.resources
from typing import TYPE_CHECKING, Union
import copy

import numpy as np
import scipy as sp

from ..dynamics import dcms
from . import propagation, logging, orbit

if TYPE_CHECKING:
    from . import orbit
    from . import time


class Perturbation(ABC):
    r"""
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
        r"""
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


# TODO: Deal with the singularity at the poles (division by a trig term which is zero at polar colatitudes).
class NonSphericalEarth(Perturbation):
    r"""
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
        r"""
        Computes the perturbing acceleration using a geopotential model of the Earth's gravitational field.

        First the colatitude and longitude are found from the current time and state using
        :meth:`compute_colat_and_long`. The, the perturbing accelerations are computed in Earth-centered Earth-fixed
        (ECEF) curvilinear/spherical coordinates. Finally, this is transformed back to rectilinear and then
        Earth-centered inertial (ECI) coordinates using DCMs generated via
        :func:`~hohmannpy.dynamics.dcms.euler_2_dcm()`.

        Parameters
        ----------
        time : float
            Current time in seconds since propagation began.
        state : np.ndarray
            Current translational state in ECI coordinates given as (position, velocity).

        Returns
        -------
        acceleration : np.ndarray
            Current translational acceleration in ECI coordinates.
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
                        * (-1) ** m * legendre_funcs[0, n, m]
                        * (self.c_coeffs[n, m] * np.cos(m * longitude) + self.s_coeffs[n, m] * np.sin(m * longitude))
                )
                longitudinal_accel += (
                    1 / (radius ** 2 * np.sin(colatitude))
                        * grav_param * (earth_radius / radius) ** n
                        * (-1) ** m * legendre_funcs[0, n, m]
                        * m
                        * (self.c_coeffs[n, m] * -np.sin(m * longitude) + self.s_coeffs[n, m] * np.cos(m * longitude))
                )
                colatitudinal_accel += (
                    -1 / radius ** 2
                        * grav_param * (earth_radius / radius) ** n
                        * (-1) ** m * legendre_funcs[1, n, m] * np.sin(colatitude)
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
        r"""
        Computes the colatitude and longitude of the satellite wrt. the Greenwich meridian from the PCI position and
        time.

        Parameters
        ----------
        time : float
            Current time in seconds since propagation began.
        position : np.ndarray
            Current ECI position.

        Returns
        -------
        colatitude : float
            Angle between the ECEF 3-axis and position vector.
        longitude : float
            Angle between the ECEF 1-axis and the projection of the position vector in to the 1-2 ECEF plane..
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


class J2(Perturbation):
    r"""
    Perturbation caused by Earth's equatorial bulge, known as the J2 effect.

    This is a simplified version of :class:`~hohmannpy.astro.perturbations.NonSphericalEarth` intended for us in
    modeling purely the J2 effect. The J2-acceleration is computed explicitly in Cartesian Earth-centered Earth-fixed
    (ECEF) coordinates before being transformed back to Earth-centered inertial coordinates (ECI). This requires use of
    the GMST to orient the Earth wrt. the inertial frame. For simplicity, the GMST is located accurately (including
    precession of the Vernal equinox) at the start of the simulation. However, for the length of propagation it is said
    to simply rotate at the Earth's mean rotation rate, ignoring precession effects.

    Parameters
    ----------
    gmst : float
        Current angle of the Greenwich meridian in :math:`rad`.

    Attributes
    ----------
    initial_gmst : float
        Initial angle of the Greenwich meridian in :math:`rad` when propagation began.

    See Also
    --------
    :class:`~hohmannpy.astro.perturbations.NonSphericalEarth` : Generalized version of this perturbation which can model
        N-order zonal harmonic effects as well as tesseral and sectoral ones.
    """

    def __init__(self, gmst):
        super().__init__()

        self.initial_gmst = gmst

    def evaluate(self, time: float, state: np.ndarray) -> tuple[float, float, float]:
        r"""
        Computes the perturbing acceleration due to the J2 effect.

        Parameters
        ----------
        time : float
            Current time in seconds since propagation began.
        state : np.ndarray
            Current translational state in ECI coordinates given as (position, velocity).

        Returns
        -------
        acceleration : np.ndarray
            Current translational acceleration in PCI coordinates.
        """

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


# TODO: Deal with the singularity at the poles (division by a trig term which is zero at polar colatitudes).
class AtmosphericDrag(Perturbation):
    r"""
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
        r"""
        Computes the perturbing acceleration using a model for the drag caused by the Earth's atmosphere.

        The geodetic altitude is first found using :meth:`compute_altitude()` assuming an ellipsoid Earth and then the
        density is found via interpolation of atmospheric data. Using this the velocity wrt. the relative wind and then
        acceleration due to drag are found.

        Parameters
        ----------
        time : float
            Current time in seconds since propagation began.
        state : np.ndarray
            Current translational state in PCI coordinates given as (position, velocity).

        Returns
        -------
        acceleration : np.ndarray
            Current translational acceleration in PCI coordinates.
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

        Parameters
        ----------
        position : np.ndarray
            Current position in PCI coordinates.

        Returns
        -------
        altitude : float
            Current height above sea level of an ellipsoid Earth.

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


class ThirdBodyGravity(Perturbation):
    r"""
    Perturbation caused by a third body's gravity. This third body can either orbit the central body or
    orbit another arbitrary fixed point.

    This class takes in a :class:`~hohmannpy.astro.Orbit` which represents the orbit of the third body. In addition, it
    optionally accepts a second orbit which represents that of the central body. This is useful in situations where both
    the central and third body orbit another object, such as two of Jupiter's moons about Jupiter. If the second
    orbit is not passed then it is assumed that the third body object simply orbits the central body. These orbits are
    then propagated during initialization. These propagated orbits are then converted to :class:`scipy.BSpline` which
    can then be used to interpolate the position of these bodies for any time lookup in :meth:`evaluate()`.

    When said method is called the relative acceleration of the central body due to the third body as well as the direct
    acceleration of the satellite due to the third body are computed.

    Parameters
    ----------
    initial_global_time: :class:`~hohmannpy.astro.Time`
        Gregorian date and UT1 time at which propagation of the third (and optionally central) body orbits should begin.
        Should match the initial and final time passed to the :class:`~hohmannpy.astro.Mission` which holds this
        perturbation.
    final_global_time: :class:`~hohmannpy.astro.Time`
        Gregorian date and UT1 time at which propagation of the third (and optionally central) body orbits should end.
        Should match the initial and final time passed to the :class:`~hohmannpy.astro.Mission` which holds this
        perturbation.
    third_body_orbit: :class:`~hohmannpy.astro.Orbit`
        Orbit of the third body.
    central_body_orbit: :class:`~hohmannpy.astro.Orbit`
        Orbit of the central body. Optional, if not provided it will be assumed the third body orbits the central body.
    propagator: :class:`~hohmannpy.astro.Propagator`
        What propagation method to use for the central and third body's orbits. These are assumed to be Keplerian and as
        such either :class:`~hohmannpy.astro.KeplerPropagator` or :class:`~hohmannpy.astro.UniversalVariablePropagator`.
    legendre: bool
        Whether to use a Legendre polynomial expansion in the computation of the third body's perturbing effects. Used
        to avoid small difference numerical accuracy losses from the difference between the two position cubics due to
        their potential similarities.
    legendre_series_length: int
        If a Legendre polynomial expansion is used, how many terms to include.

    Attributes
    ----------
    tb_grav_param : float
        Gravitational parameter of the third body.
    legendre: bool
        Whether to use a Legendre polynomial expansion in the computation of the third body's perturbing effects. Used
        to avoid small difference numerical accuracy losses from the difference between the two position cubics due to
        their potential similarities.
    legendre_series_length: int
        If a Legendre polynomial expansion is used, how many terms to include.
    tb_orbit_spline : :class:`scipy.BSpline`
        Linear spline of the third body trajectory. Calling it via ``tb_orbit_spline(time)`` returns the interpolated
        orbit at that time.
    cb_orbit_spline : Union[:class:`scipy.BSpline`, func]
        Linear spline of the central body trajectory. Calling it via ``tb_orbit_spline(time)`` returns the interpolated
        orbit at that time.
    """

    def __init__(
            self,
            initial_global_time: time.Time,
            final_global_time: time.Time,
            third_body_orbit: orbit.Orbit,
            central_body_orbit: orbit.Orbit = None,
            propagator: propagation.Propagator = propagation.UniversalVariablePropagator(),
            legendre: bool = True,
            legendre_series_length: int = 10,
    ):
        super().__init__()

        self.tb_grav_param = third_body_orbit.grav_param
        self.legendre = legendre
        self.legendre_series_length = legendre_series_length

        # Safeguard to make sure the propagator has a state logger because we need this.
        if not any(isinstance(logger, logging.StateLogger) for logger in propagator.loggers):
            propagator.loggers.insert(0, logging.StateLogger())
        tb_propagator = copy.deepcopy(propagator)
        cb_propagator = propagator

        # Setup propagator and then call propagate() to generate trajectory of third-body. Then convert this to a
        # numpy.BSpline.
        tb_propagator.setup(
            orbit=third_body_orbit,
            final_time=(final_global_time.julian_date - initial_global_time.julian_date) * 86400
        )
        tb_propagator.propagate()
        tb_times = tb_propagator.loggers[0].time_history
        tb_traj = tb_propagator.loggers[0].position_history
        self.tb_orbit_spline = sp.interpolate.make_interp_spline(tb_times.squeeze(), tb_traj.T, k=1)

        # Setup propagator and then call propagate() to generate trajectory of the central-body. Then convert this to a
        # numpy.BSpline.
        if central_body_orbit is not None:
            cb_propagator.setup(
                orbit=central_body_orbit,
                final_time=(final_global_time.julian_date - initial_global_time.julian_date) * 86400
            )
            cb_propagator.propagate()
            cb_times = cb_propagator.loggers[0].time_history
            cb_traj = cb_propagator.loggers[0].position_history
            self.cb_orbit_spline = sp.interpolate.make_interp_spline(cb_times.squeeze(), cb_traj.T, k=1)
        else:  # If no orbit provide assume central body is stationary (ex. third body is a moon).
            def dummy_spline(x):
                return np.array([0, 0, 0])
            self.cb_orbit_spline = dummy_spline

    def evaluate(self, time: float, state: np.ndarray) -> tuple[float, float, float]:
        """
        Computes the perturbing acceleration due to the third body.

        Uses :attr:`tb_orbit_spline` and :attr:`cb_orbit_spline` to approximate the position of the third body wrt. the
        central body and satellite. These are then used in conjunction with Newton's Law of Universal Gravitation to
        compute the total perturbing acceleration due to the third body. Optionally, some of the terms in this equation
        are approximated using Legendre polynomials to avoid numerical difficulties.

        Parameters
        ----------
        time : float
            Current time in seconds since propagation began.
        state : np.ndarray
            Current translational state in ECI coordinates given as (position, velocity).

        Returns
        -------
        acceleration : np.ndarray
            Current translational acceleration in PCI coordinates.
        """

        # Calculate position vectors.
        position_tb_wrt_cb = -self.cb_orbit_spline(time) + self.tb_orbit_spline(time)
        position_tb_wrt_sat = position_tb_wrt_cb - state[:3]

        if self.legendre:
            # Compute cosine of phase angle.
            phase_angle_cosine = (
                    np.dot(state[:3], position_tb_wrt_cb)
                        / (np.linalg.norm(state[:3]) * np.linalg.norm(position_tb_wrt_cb))
            )

            # Get sum of Legendre polynomials.
            legendre_sum = 0
            position_ratio = np.linalg.norm(state[:3]) / np.linalg.norm(position_tb_wrt_cb)
            for i in range(1, self.legendre_series_length):
                legendre_sum += sp.special.legendre_p(i, phase_angle_cosine) * position_ratio ** i

            # Compute acceleration.
            acceleration = (
                    -self.tb_grav_param / np.linalg.norm(position_tb_wrt_cb) ** 3
                        * (state[:3] - position_tb_wrt_sat * (3 * legendre_sum + 3 * legendre_sum ** 2 + legendre_sum ** 3))
            )
        else:
            acceleration = (
                self.tb_grav_param * (
                    position_tb_wrt_sat / np.linalg.norm(position_tb_wrt_sat) ** 3
                        - position_tb_wrt_cb / np.linalg.norm(position_tb_wrt_cb) ** 3
                )
            )

        return acceleration[0], acceleration[1], acceleration[2]


class LunarGravity(ThirdBodyGravity):
    r"""
    Perturbation caused by the Moon's gravity.

    This class implements a specialized version of :class:`~hohmannpy.astro.perturbations.ThirdBodyGravity` adjusted to
    specifically account for the third body perturbations due to the Earth's moon.

    Note that the true anomaly of the Moon is not computed automatically based on the dates of desired propagation, the
    accurate true anomaly must be input separately. This is because the orbit of the moon is not truly Keplerian due to
    apsidal precession amongst other effects. Another result of this is that the Keplerian propagation of the Moon's
    orbit used by this class for simulating its third-body gravitational effects is also coarse.

    Parameters
    ----------
    initial_global_time: :class:`~hohmannpy.astro.Time`
        Gregorian date and UT1 time at which propagation of the third (and optionally central) body orbits should begin.
        Should match the initial and final time passed to the :class:`~hohmannpy.astro.Mission` which holds this
        perturbation.
    final_global_time: :class:`~hohmannpy.astro.Time`
        Gregorian date and UT1 time at which propagation of the third (and optionally central) body orbits should end.
        Should match the initial and final time passed to the :class:`~hohmannpy.astro.Mission` which holds this
        perturbation.
    initial_true_anomaly: float
        True anomaly of the Moon.
    propagator: :class:`~hohmannpy.astro.propagation.Propagator`
        What propagation method to use for the central and third body's orbits. These are assumed to be Keplerian and as
        such either :class:`~hohmannpy.astro.propagation.KeplerPropagator` or
        :class:`~hohmannpy.astro.propagation.UniversalVariablePropagator`.
    legendre: bool
        Whether to use a Legendre polynomial expansion in the computation of the third body's perturbing effects. Used
        to avoid small difference numerical accuracy losses from the difference between the two position cubics due to
        their potential similarities.
    legendre_series_length: int
        If a Legendre polynomial expansion is used, how many terms to include.

    See Also
    --------
    :class:`~hohmannpy.astro.perturbations.ThirdBodyGravity` : Base version of this class which can be used for any third body.
    """

    def __init__(
            self,
            initial_global_time: time.Time,
            final_global_time: time.Time,
            initial_true_anomaly: float,
            propagator: propagation.Propagator = propagation.UniversalVariablePropagator(),
            legendre: bool = True,
            legendre_series_length: int = 10,
    ):
        lunar_orbit = orbit.Orbit.from_classical_elements(
            sm_axis=3.844e8,
            eccentricity=0.0549,
            inclination=np.rad2deg(5.145),
            raan=np.rad2deg(125.08),
            argp=np.deg2rad(318.15),
            true_anomaly=initial_true_anomaly,
        )

        super().__init__(
            initial_global_time=initial_global_time,
            final_global_time=final_global_time,
            third_body_orbit=lunar_orbit,
            propagator=propagator,
            legendre=legendre,
            legendre_series_length=legendre_series_length
        )


class SolarGravity(ThirdBodyGravity):
    r"""
    Perturbation caused by the Sun's gravity.

    This class implements a specialized version of :class:`~hohmannpy.astro.perturbations.ThirdBodyGravity` adjusted to
    specifically account for the third body perturbations due to the Sun. Note that the true anomaly of the Earth wrt.
    to the ecliptic plane is computed automatically based on the dates of desired propagation.

    Parameters
    ----------
    initial_global_time: :class:`~hohmannpy.astro.Time`
        Gregorian date and UT1 time at which propagation of the third (and optionally central) body orbits should begin.
        Should match the initial and final time passed to the :class:`~hohmannpy.astro.Mission` which holds this
        perturbation.
    final_global_time: :class:`~hohmannpy.astro.Time`
        Gregorian date and UT1 time at which propagation of the third (and optionally central) body orbits should end.
        Should match the initial and final time passed to the :class:`~hohmannpy.astro.Mission` which holds this
        perturbation.
    propagator: :class:`~hohmannpy.astro.propagation.Propagator`
        What propagation method to use for the central and third body's orbits. These are assumed to be Keplerian and as
        such either :class:`~hohmannpy.astro.propagation.KeplerPropagator` or
        :class:`~hohmannpy.astro.propagation.UniversalVariablePropagator`.
    legendre: bool
        Whether to use a Legendre polynomial expansion in the computation of the third body's perturbing effects. Used
        to avoid small difference numerical accuracy losses from the difference between the two position cubics due to
        their potential similarities.
    legendre_series_length: int
        If a Legendre polynomial expansion is used, how many terms to include.

    See Also
    --------
    :class:`~hohmannpy.astro.perturbations.ThirdBodyGravity` : Base version of this class which can be used for any third body.
    """

    def __init__(
            self,
            initial_global_time: time.Time,
            final_global_time: time.Time,
            solver_tol: float = 1e-8,
            propagator: propagation.Propagator = propagation.UniversalVariablePropagator(),
            legendre: bool = True,
            legendre_series_length: int = 10,
    ):
        initial_true_anomaly = self.compute_initial_true_anomaly(initial_global_time, solver_tol)
        earth_orbit = orbit.Orbit.from_classical_elements(
            sm_axis=149597870.7e3,
            eccentricity=0.0167086,
            inclination=0,
            raan=0,
            argp=np.deg2rad(102.937),
            true_anomaly=initial_true_anomaly,
            grav_param=1.32712440018e20
        )

        super().__init__(
            initial_global_time=initial_global_time,
            final_global_time=final_global_time,
            third_body_orbit=earth_orbit,
            propagator=propagator,
            legendre=legendre,
            legendre_series_length=legendre_series_length
        )

        # Change from Earth orbiting Earth to Sun orbiting Earth.
        tb_orbit_spline = copy.deepcopy(self.tb_orbit_spline)
        def inverted_spline(x):
            return tb_orbit_spline(x) * -1
        self.tb_orbit_spline = inverted_spline

    def compute_initial_true_anomaly(self, initial_global_time: time.Time, solver_tol: float):
        """
        Calculates the true anomaly of the Earth at the initial date.

        Parameters
        ----------
        initial_global_time : time.Time
            Gregorian date and UT1 time at which the Earth is initially located.
        solver_tol : float
            Error tolerance to use when solving Kepler's equation.

        Returns
        -------
        initial_true_anomaly : float
            True anomaly of the earth corresponding to ``initial_global_time``.
        """

        earth_mean_motion = np.deg2rad(0.98560028)
        earth_eccentricity = 0.0167086
        j2000_mean_anomaly = 357.5277233
        j2000_julian_time = 2451545

        # Compute the initial mean anomaly wrt. J2000 and then solve Kepler's equation for the corresponding initial
        # eccentric anomaly.
        initial_mean_anomaly = (
                j2000_mean_anomaly
                    + earth_mean_motion * (initial_global_time.julian_date - j2000_julian_time)
        )
        eq = lambda x: initial_mean_anomaly - x + earth_eccentricity * np.sin(x)
        initial_eccentric_anomaly = sp.optimize.newton(eq, 0, tol=solver_tol)

        # Use Gauss' equation to compute the initial eccentric anomaly to the initial true anomaly.s
        return  (
            2 * np.arctan(
                np.sqrt((1 + earth_eccentricity) / (1 - earth_eccentricity)) * np.tan(initial_eccentric_anomaly / 2)
            )
        )


class SolarRadiation(Perturbation):
    def __init__(
            self,
            solar_area: float,
            reflectivity: float,
            initial_global_time: time.Time,
            final_global_time: time.Time,
            propagator: propagation.Propagator = propagation.UniversalVariablePropagator(),

    ):
        super().__init__()

        initial_true_anomaly = self.compute_initial_true_anomaly(initial_global_time, solver_tol)
        earth_orbit = orbit.Orbit.from_classical_elements(
            sm_axis=149597870.7e3,
            eccentricity=0.0167086,
            inclination=0,
            raan=0,
            argp=np.deg2rad(102.937),
            true_anomaly=initial_true_anomaly,
            grav_param=1.32712440018e20
        )

    @abstractmethod
    def evaluate(self, time: float, state: np.ndarray) -> tuple[float, float, float]:
        r"""
        TBD

        Parameters
        ----------
        time : float
            Current time in seconds since propagation began.
        state : np.ndarray
            Current translational state in planet-centered inertial coordinates given as (position, velocity).
        """

        pass

    def compute_initial_true_anomaly(self, initial_global_time: time.Time, solver_tol: float):
        """
        Calculates the true anomaly of the Earth at the initial date.

        Parameters
        ----------
        initial_global_time : time.Time
            Gregorian date and UT1 time at which the Earth is initially located.
        solver_tol : float
            Error tolerance to use when solving Kepler's equation.

        Returns
        -------
        initial_true_anomaly : float
            True anomaly of the earth corresponding to ``initial_global_time``.
        """

        earth_mean_motion = np.deg2rad(0.98560028)
        earth_eccentricity = 0.0167086
        j2000_mean_anomaly = 357.5277233
        j2000_julian_time = 2451545

        # Compute the initial mean anomaly wrt. J2000 and then solve Kepler's equation for the corresponding initial
        # eccentric anomaly.
        initial_mean_anomaly = (
                j2000_mean_anomaly
                + earth_mean_motion * (initial_global_time.julian_date - j2000_julian_time)
        )
        eq = lambda x: initial_mean_anomaly - x + earth_eccentricity * np.sin(x)
        initial_eccentric_anomaly = sp.optimize.newton(eq, 0, tol=solver_tol)

        # Use Gauss' equation to compute the initial eccentric anomaly to the initial true anomaly.s
        return (
                2 * np.arctan(
            np.sqrt((1 + earth_eccentricity) / (1 - earth_eccentricity)) * np.tan(initial_eccentric_anomaly / 2)
        )
        )
