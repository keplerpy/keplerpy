from __future__ import annotations
from abc import ABC, abstractmethod

import numpy as np

from . import propagation


# TODO:
#  - Add labels to all these variables for when they are stored in a .csv.
#  - Docstrings for the child classes.
class Logger(ABC):
    r"""
    A logger is used to store data regarding an orbit generated on each timestep by :class:`~hohmannpy.astro.Propagator`
    . :meth:`~hohmannpy.astro.Propagator.propagate()`.

    This is done by inserting values into a set of ``np.ndarray``'s each of size (M, N) where M is the length of the
    value being logged (if it's a vector) and N is the number of timesteps propagation, and hence logging, occurs for at
    each timestep.

    One or more loggers are passed to a propagator during startup. However, the ``__init__()`` for each logger is empty
    as the necessary information (such as the initial value of each variable being logged on the first timestep) to
    begin logging is not known until :class:`~hohmannpy.astro.Propagator` . :meth:`~hohmannpy.astro.Propagator.setup()`
    is called. At this point the local :meth:`setup()` is also called to prepare the logger for logging. Then, as the
    propagator steps through the propagation timesteps :meth:`log` is called on each of them to perform the array
    indexing mentioned previously.
    """

    def __init__(self):
        pass

    @abstractmethod
    def setup(self, propagator: propagation.base.Propagator):
        r"""
        Sets up a propagator.

        All child classes must implement this method with the following steps:

        1) Allocate space using :func:`numpy.zeros()` where the data is stored column-wise with N columns where N = the number of timesteps stored in the ``Propagator``'s ``timestep`` attribute.

        2) Fill in the 0th column of each array with the orbit's initial values for the stored data.

        Parameters
        ----------
        propagator : :class:`~hohmannpy.astro.Propagator`
            The propagator which passes in data for logging.

        Notes
        -----
        Can't call this till after the initial values of propagator-specific attributes, such as ``eccentric_anomaly``
        for :class:`~hohmannpy.astro.KeplerPropagator`, have been set. This is typically towards the start of a
        propagators' ``propagate()`` method.
        """

        pass

    @abstractmethod
    def log(self, propagator: propagation.base.Propagator, timestep: int):
        r"""
        Fills in the Nth column of each history array with the orbit's current values for each data. The data is
        accessed from the propagator's ``orbit`` attribute which is itself an instance of :class:`~hohmannpy.astro.Orbit`.

        Parameters
        ----------
        propagator : :class:`~hohmannpy.astro.Propagator`
            The propagator which passes in data for logging.
        timestep: int
            How many timesteps propagation has occurred for. Used to row-index the history arrays.
        """

        pass

    @abstractmethod
    def to_csv(self):
        r"""
        Converts all the history arrays to a human-readable CSV file.
        """

        pass


class StateLogger(Logger):
    """
    Logs the time and Cartesian state (position and velocity) of the orbit.
    """
    def __init__(self):
        super().__init__()

        self.position_history = None
        self.velocity_history = None
        self.time_history = None

        self.labels = [
            "Time (s)",
            "x-Position (km)", "y-Position (km)", "z-Position (km)",
            "x-Velocity (km/s)", "y-Velocity (km/s)", "z-Velocity (km/s)",
        ]

    def setup(self, propagator: propagation.base.Propagator):
        self.position_history = np.zeros([3, propagator.timesteps + 1])
        self.velocity_history = np.zeros([3, propagator.timesteps + 1])
        self.time_history = np.zeros([1, propagator.timesteps + 1])

        self.position_history[:, 0] = propagator.orbit.position
        self.velocity_history[:, 0] = propagator.orbit.velocity
        self.time_history[0, 0] = propagator.orbit.time

    def log(self, propagator: propagation.base.Propagator, timestep: int):
        self.position_history[:, timestep] = propagator.orbit.position
        self.velocity_history[:, timestep] = propagator.orbit.velocity
        self.time_history[0, timestep] = propagator.orbit.time

    def to_csv(self):
        pass


class ElementsLogger(Logger):
    """
    Logs the orbital elements of the orbit.
    """

    def __init__(self):
        self.sm_axis_history = None
        self.eccentricity_history = None
        self.inclination_history = None
        self.raan_history = None
        self.argp_history = None
        self.true_anomaly_history = None
        self.longp_history = None
        self.argl_history = None
        self.true_latitude_history = None
        self.e_component1_history = None
        self.e_component2_history = None
        self.n_component1_history = None
        self.n_component2_history = None
        super().__init__()

    def setup(self, propagator: propagation.base.Propagator):
        self.sm_axis_history = np.zeros([1, propagator.timesteps + 1])
        self.eccentricity_history = np.zeros([1, propagator.timesteps + 1])
        self.inclination_history = np.zeros([1, propagator.timesteps + 1])
        self.raan_history = np.zeros([1, propagator.timesteps + 1])
        self.argp_history = np.zeros([1, propagator.timesteps + 1])
        self.true_anomaly_history = np.zeros([1, propagator.timesteps + 1])
        self.longp_history = np.zeros([1, propagator.timesteps + 1])
        self.argl_history = np.zeros([1, propagator.timesteps + 1])
        self.true_latitude_history = np.zeros([1, propagator.timesteps + 1])

        self.sm_axis_history[0, 0] = propagator.orbit.sm_axis
        self.eccentricity_history[0, 0] = propagator.orbit.eccentricity
        self.inclination_history[0, 0] = propagator.orbit.inclination
        self.raan_history[0, 0] = propagator.orbit.raan
        self.argp_history[0, 0] = propagator.orbit.argp
        self.true_anomaly_history[0, 0] = propagator.orbit.true_anomaly
        self.longp_history[0, 0] = propagator.orbit.longp
        self.argl_history[0, 0] = propagator.orbit.argl
        self.true_latitude_history[0, 0] = propagator.orbit.true_latitude

        if propagator.orbit.track_equinoctial:
            self.e_component1_history = np.zeros([1, propagator.timesteps + 1])
            self.e_component2_history = np.zeros([1, propagator.timesteps + 1])
            self.n_component1_history = np.zeros([1, propagator.timesteps + 1])
            self.n_component2_history = np.zeros([1, propagator.timesteps + 1])

            self.e_component1_history[0, 0] = propagator.orbit.e_component1
            self.e_component2_history[0, 0] = propagator.orbit.e_component2
            self.n_component1_history[0, 0] = propagator.orbit.n_component1
            self.n_component2_history[0, 0] = propagator.orbit.n_component2

    def log(self, propagator: propagation.base.Propagator, timestep: int):
        self.sm_axis_history[0, timestep] = propagator.orbit.sm_axis
        self.eccentricity_history[0, timestep] = propagator.orbit.eccentricity
        self.inclination_history[0, timestep] = propagator.orbit.inclination
        self.raan_history[0, timestep] = propagator.orbit.raan
        self.argp_history[0, timestep] = propagator.orbit.argp
        self.true_anomaly_history[0, timestep] = propagator.orbit.true_anomaly
        self.longp_history[0, timestep] = propagator.orbit.longp
        self.argl_history[0, timestep] = propagator.orbit.argl
        self.true_latitude_history[0, timestep] = propagator.orbit.true_latitude

        if propagator.orbit.track_equinoctial:
            self.e_component1_history[0, timestep] = propagator.orbit.e_component1
            self.e_component2_history[0, timestep] = propagator.orbit.e_component2
            self.n_component1_history[0, timestep] = propagator.orbit.n_component1
            self.n_component2_history[0, timestep] = propagator.orbit.n_component2

class EccentricAnomalyLogger(Logger):
    """
    Logs the eccentric anomaly of an orbit (for use with KeplerPropagator).
    """

    def __init__(self):
        self.eccentric_anomaly_history = None
        super().__init__()

    def setup(self, propagator: propagation.kepler.KeplerPropagator):
        self.eccentric_anomaly_history = np.zeros([1, propagator.timesteps + 1])

        self.eccentric_anomaly_history[0, 0] = propagator.eccentric_anomaly

    def log(self, propagator: propagation.kepler.KeplerPropagator, timestep: int):
        self.eccentric_anomaly_history[0, timestep] = propagator.eccentric_anomaly


class UniversalVariableLogger(Logger):
    """
    Logs the universal variable and Stumpff parameter (for use with UniversalVariablePropagator).
    """

    def __init__(self):
        self.universal_variable_history = None
        self.stumpff_param_history = None
        super().__init__()

    def setup(self, propagator: propagation.universal_variable.UniversalVariablePropagator):
        self.universal_variable_history = np.zeros([1, propagator.timesteps + 1])
        self.stumpff_param_history = np.zeros([1, propagator.timesteps + 1])

        self.universal_variable_history[0, 0] = propagator.universal_variable
        self.stumpff_param_history[0, 0] = propagator.stumpff_param

    def log(self, propagator: propagation.universal_variable.UniversalVariablePropagator, timestep: int):
        self.universal_variable_history[0, timestep] = propagator.universal_variable
        self.stumpff_param_history[0, timestep] = propagator.stumpff_param
