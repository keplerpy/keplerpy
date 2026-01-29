from __future__ import annotations
import importlib.resources

import numpy as np
import scipy as sp

from ...dynamics import dcms
from . import base


# TODO: Deal with the singularity at the poles (division by a trig term which is zero at polar colatitudes).
class NonSphericalEarth(base.Perturbation):
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
        acceleration : tuple[float, float, float]
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
        Computes the colatitude and longitude of the satellite wrt. the Greenwich meridian from the ECI position and
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


class J2(base.Perturbation):
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
        acceleration : tuple[float, float, float]
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
