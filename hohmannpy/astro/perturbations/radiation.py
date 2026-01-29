from __future__ import annotations

import numpy as np
import scipy as sp

from ...dynamics import dcms
from .. import propagation, logging, time, orbit
from . import base


class SolarRadiation(base.Perturbation):
    r"""
    Perturbation caused by solar radiation from the Sun.

    This class functions similar to :class:`~hohmannpy.astro.perturbations.SolarGravity`. During initialization the
    orbit of the Earth (assumed to be Keplerian) is propagated and then converted into a :class:`numpy.BSpline` which
    can then be used for interpolation during calls to :meth:`evaluate()` when computing the position of the sun wrt.
    the satellite.

    A few assumptions are made to simply computation. First, the solar irradiance is given by an equation developed by
    Wertz [1]. This model uses the number of days passed since aphelion. Aphelion is taken to have occurred on the most
    recent July 4th, 12:00:00 UT1 preceding ``initial_global_time``. It is then taken to occur every 365.25 Julian days
    after this (leap days are not accounted for). The 1-2 plane of the Earth-centered-inertial basis is also assumed to
    be inclined at constant 23.5 :math:`deg` from the ecliptic plane. The reflective area facing the Sun and
    reflectivity of said area is also assumed to be constant with the satellite's attitude not accounted for. As a
    result, the perturbing acceleration is said to act along a line from the Sun to the satellite. Finally, shade due to
    the Earth is not accounted for which will introduce inaccuracies for LEO orbits.

    Parameters
    ----------
    mean_reflective_area : float
        Average reflective area exposed to the sun.
    reflectivity : float
        Constant representing the reflectivity of the satellite. 0 = transparent, 1 = full absorption, and 2 = full
        reflection.
    mass : float
        Average mass of the satellite.
    initial_global_time: :class:`~hohmannpy.astro.Time`
        Gregorian date and UT1 time at which propagation of the third (and optionally central) body orbits should begin.
        Should match the initial and final time passed to the :class:`~hohmannpy.astro.Mission` which holds this
        perturbation.
    final_global_time: :class:`~hohmannpy.astro.Time`
        Gregorian date and UT1 time at which propagation of the third (and optionally central) body orbits should end.
        Should match the initial and final time passed to the :class:`~hohmannpy.astro.Mission` which holds this
        perturbation.
    irradiance_scale_factor : float
        Constant to scale the solar irradiance by at all timesteps. Useful for representing heightened solar activity
        such as during solar flares.
    propagator: :class:`~hohmannpy.astro.propagation.Propagator`
        What propagation method to use for the Earth's orbit. This is assumed to be Keplerian and as such either
        :class:`~hohmannpy.astro.propagation.KeplerPropagator` or
        :class:`~hohmannpy.astro.propagation.UniversalVariablePropagator` must be used.
    solver_tol : float
        Error tolerance to use when solving Kepler's equation for the Earth's initial true anomaly.

    Attributes
    ----------
    mean_reflective_area : float
        Average reflective area exposed to the sun.
    reflectivity : float
        Constant representing the reflectivity of the satellite. 0 = transparent, 1 = full absorption, and 2 = full
        reflection.
    mass : float
        Average mass of the satellite.
    irradiance_scale_factor : float
        Constant to scale the solar irradiance by at all timesteps. Useful for representing heightened solar activity
        such as during solar flares.
    earth_orbit_spline : :class:`numpy.BSpline`
        Linear spline of the Earth's trajectory. Calling it via ``earth_orbit_spline(time)`` returns the interpolated
        orbit at that time.
    initial_jd_since_aphelion : float
        Number of Julian days passed since aphelion, taken to be on the most recent July 4th before
        ``initial_global_time``.

    .. [1] James R. Wertz, Spacecraft Attitude Determination and Control, Astrophysics and Space Science Library, vol.
        73. Dordrecht, The Netherlands: Springer, 1978
    """

    def __init__(
            self,
            mean_reflective_area: float,
            reflectivity: float,
            mass: float,
            initial_global_time: time.Time,
            final_global_time: time.Time,
            irradiance_scale_factor: float = 1,
            propagator: propagation.Propagator = propagation.UniversalVariablePropagator(),
            solver_tol: float = 1e-8,

    ):
        super().__init__()

        self.mean_reflective_area = mean_reflective_area
        self.reflectivity = reflectivity
        self.mass = mass
        self.irradiance_scale_factor = irradiance_scale_factor

        # Initialize Earth's orbit.
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

        # Safeguard to make sure the propagator has a state logger because we need this.
        if not any(isinstance(logger, logging.StateLogger) for logger in propagator.loggers):
            propagator.loggers.insert(0, logging.StateLogger())

        # Setup propagator and then call propagate() to generate the Earth's trajectory. Then convert this to a
        # numpy.BSpline.
        propagator.setup(
            orbit=earth_orbit,
            final_time=(final_global_time.julian_date - initial_global_time.julian_date) * 86400
        )
        propagator.propagate()
        earth_times = propagator.loggers[0].time_history
        earth_traj = propagator.loggers[0].position_history
        self.earth_orbit_spline = sp.interpolate.make_interp_spline(earth_times.squeeze(), earth_traj.T, k=1)

        # Compute the Julian days since the most recent aphelion.
        initial_date = initial_global_time.date
        initial_month = initial_date[3:5]

        if int(initial_month) > 7:
            initial_year = int(initial_date[6:])
        else:
            initial_year = int(initial_date[6:]) - 1

        aphelion_time = time.Time(date=f"07/04/{initial_year}", time="12:00:00")
        self.initial_jd_since_aphelion = initial_global_time.julian_date - aphelion_time.julian_date

    def evaluate(self, time: float, state: np.ndarray) -> tuple[float, float, float]:
        r"""
        Computes the perturbing acceleration due to the Sun's radiation.

        First queries :attr:`earth_orbit_spline` to get the position of the Earth wrt. the Sun and then uses that to
        construct the position of the Sun wrt. the satellite. Then computes the solar pressure and using that the
        acceleration is calculated.

        Parameters
        ----------
        time : float
            Current time in seconds since propagation began.
        state : np.ndarray
            Current translational state in planet-centered inertial (PCI) coordinates given as (position, velocity).

        Returns
        -------
        acceleration : tuple[float, float, float]
            Current translational acceleration in PCI coordinates.
        """

        speed_of_light = 3e8

        # Compute the position of the Sun wrt. the satellite.
        earth_tilt = np.deg2rad(-23.439291115)
        position_earth_wrt_sun = dcms.euler_2_dcm(earth_tilt, 1) @ self.earth_orbit_spline(time)
        position_sun_wrt_sat = -(position_earth_wrt_sun + state[:3])

        # Compute the solar pressure.
        days_since_aphelion = (self.initial_jd_since_aphelion + time / 86400) % 365.25
        irradiance = 1358 / (1.004 + 0.0334 * np.cos(2 * np.pi * days_since_aphelion)) * self.irradiance_scale_factor
        solar_pressure = irradiance / speed_of_light

        # Compute the acceleration.
        acceleration = (
            -solar_pressure * self.reflectivity * self.mean_reflective_area / self.mass
                * position_sun_wrt_sat / np.linalg.norm(position_sun_wrt_sat)
        )

        return acceleration[0], acceleration[1], acceleration[2]

    def compute_initial_true_anomaly(self, initial_global_time: time.Time, solver_tol: float):
        r"""
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
        j2000_mean_anomaly = np.deg2rad(357.5277233)
        j2000_julian_time = 2451545

        # Compute the initial mean anomaly wrt. J2000 and then solve Kepler's equation for the corresponding initial
        # eccentric anomaly.
        initial_mean_anomaly = (
                j2000_mean_anomaly
                + earth_mean_motion * ((initial_global_time.julian_date - j2000_julian_time) * 86400)
        ) % 2 * np.pi
        eq = lambda x: initial_mean_anomaly - x + earth_eccentricity * np.sin(x)
        initial_eccentric_anomaly = sp.optimize.newton(eq, initial_mean_anomaly, tol=solver_tol)

        # Use Gauss' equation to compute the initial eccentric anomaly to the initial true anomaly.
        return (
                2 * np.arctan(
            np.sqrt((1 + earth_eccentricity) / (1 - earth_eccentricity)) * np.tan(initial_eccentric_anomaly / 2)
            )
        )
