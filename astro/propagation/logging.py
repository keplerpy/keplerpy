from abc import ABC, abstractmethod
import numpy as np

# TYPE CHECKING ONLY IMPORTS.
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from . import base, kepler, universal_variable


class Logger(ABC):
    """
    Base class for all loggers. A Logger is meant to be instantiated in the setup() method of a Propagator and is called
    every iteration of a Propagator's propagate() loop to store data regarding the orbit being propagated.
    """

    @abstractmethod
    def setup(self, propagator: "base.Propagator"):
        """
        Equivalent to an __init__() but the former is not used because we want to be able to pass a Logger into a
        Propagator during the latter's __init__(). All child classes must implement this method with the following
        steps:
            1) Allocate space using np.zeros() where the data is stored column-wise with N columns where N = the number
               of timesteps stored in the Propagator's timestep attribute.
            2) Fill in the 0th column of each array with the orbit's initial values for the stored data.

        NOTE: Can't call this till after the initial values of Propagator-specific attributes, such as eccentric_anomaly
        for KeplerPropagator, have been set. This is typically towards the start of a Propagator's propagate() method.

        :param propagator: Propagator object which contains the Orbit object to receive data from.
        """

        pass

    @abstractmethod
    def log(self, propagator: "base.Propagator", timestep: int):
        """
        Fill in the Nth column of each history array with the orbit's current values for each data. The data is accessed
        by calling propagator.orbit.

        :param propagator: Propagator object which contains the Orbit object to receive data from.
        :param timestep: How many timesteps propagation has occurred for.
        """

        pass


class StateLogger(Logger):
    """
    Logs the time and Cartesian state (position and velocity) of the orbit.
    """

    def setup(self, propagator: "base.Propagator"):
        self.position_history = np.zeros([3, propagator.timesteps + 1])
        self.velocity_history = np.zeros([3, propagator.timesteps + 1])
        self.time_history = np.zeros([1, propagator.timesteps + 1])

        self.position_history[:, 0] = propagator.orbit.position
        self.velocity_history[:, 0] = propagator.orbit.velocity
        self.time_history[0, 0] = propagator.orbit.time

    def log(self, propagator: "base.Propagator", timestep: int):
        self.position_history[:, timestep] = propagator.orbit.position
        self.velocity_history[:, timestep] = propagator.orbit.velocity
        self.time_history[0, timestep] = propagator.orbit.time


class ElementsLogger(Logger):
    """
    Logs the classical orbital elements of the orbit.
    """

    def setup(self, propagator: "base.Propagator"):
        self.sm_axis_history = np.zeros([1, propagator.timesteps + 1])
        self.eccentricity_history = np.zeros([1, propagator.timesteps + 1])
        self.inclination_history = np.zeros([1, propagator.timesteps + 1])
        self.raan_history = np.zeros([1, propagator.timesteps + 1])
        self.argp_history = np.zeros([1, propagator.timesteps + 1])
        self.true_anomaly_history = np.zeros([1, propagator.timesteps + 1])

        self.sm_axis_history[0, 0] = propagator.orbit.sm_axis
        self.eccentricity_history[0, 0] = propagator.orbit.eccentricity
        self.inclination_history[0, 0] = propagator.orbit.inclination
        self.raan_history[0, 0] = propagator.orbit.raan
        self.argp_history[0, 0] = propagator.orbit.argp
        self.true_anomaly_history[0, 0] = propagator.orbit.true_anomaly

    def log(self, propagator: "base.Propagator", timestep: int):
        self.sm_axis_history[0, timestep] = propagator.orbit.sm_axis
        self.eccentricity_history[0, timestep] = propagator.orbit.eccentricity
        self.inclination_history[0, timestep] = propagator.orbit.inclination
        self.raan_history[0, timestep] = propagator.orbit.raan
        self.argp_history[0, timestep] = propagator.orbit.argp
        self.true_anomaly_history[0, timestep] = propagator.orbit.true_anomaly

class EccentricAnomalyLogger(Logger):
    """
    Logs the eccentric anomaly of an orbit (for use with KeplerPropagator).
    """

    def setup(self, propagator: "kepler.KeplerPropagator"):
        # if not isinstance(propagator, kepler.KeplerPropagator):  # Safeguard.
        #     raise TypeError("Propagator must be of type KeplerPropagator to use the EccentricAnomalyLogger.")

        self.eccentric_anomaly_history = np.zeros([1, propagator.timesteps + 1])

        self.eccentric_anomaly_history[0, 0] = propagator.eccentric_anomaly

    def log(self, propagator: "kepler.KeplerPropagator", timestep: int):
        self.eccentric_anomaly_history[0, timestep] = propagator.eccentric_anomaly


class UniversalVariableLogger(Logger):
    """
    Logs the universal variable and Stumpff parameter (for use with UniversalVariablePropagator).
    """

    def setup(self, propagator: "universal_variable.UniversalVariablePropagator"):
        # if not isinstance(propagator, universal_variable.UniversalVariablePropagator):  # Safeguard.
        #     raise TypeError(
        #         "Propagator must be of type UniversalVariablePropagator to use the UniversalVariableLogger."
        #     )

        self.universal_variable_history = np.zeros([1, propagator.timesteps + 1])
        self.stumpff_param_history = np.zeros([1, propagator.timesteps + 1])

        self.universal_variable_history[0, 0] = propagator.universal_variable
        self.stumpff_param_history[0, 0] = propagator.stumpff_param

    def log(self, propagator: "universal_variable.UniversalVariablePropagator", timestep: int):
        self.universal_variable_history[0, timestep] = propagator.universal_variable
        self.stumpff_param_history[0, timestep] = propagator.stumpff_param
