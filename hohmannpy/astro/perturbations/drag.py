from __future__ import annotations
import importlib.resources

import numpy as np
import scipy as sp

from ...dynamics import dcms
from . import base


# TODO: Deal with the singularity at the poles (division by a trig term which is zero at polar colatitudes).
class AtmosphericDrag(base.Perturbation):
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
        acceleration : tuple[float, float, float]
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
