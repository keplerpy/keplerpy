from . import orbits
import numpy as np
import scipy as sp
from abc import ABC, abstractmethod


class Propagator(ABC):
    """
    Base class for all propagators. All derivatives revolve around the method propagate() which takes in the initial
    orbital parameters (whatever those might be) and propagates them along the orbit up to the final time.

    :ivar orbit: Orbit to perform propagation on. Holds the initial conditions of the orbit including the starting time.
    :ivar final time: When to stop orbit propagation.
    :ivar step_size: Time step size for propagation.

    :ivar position_history: List of orbital positions (3, ) as a (3, length of propagation) array.
    :ivar velocity_history: List of orbital velocities (3, ) as a (3, length of propagation) array.
    :ivar time_history: List of time steps as a (1, length of propagation) array.
    """

    def __init__(self, orbit, final_time, step_size):
        self.orbit = orbit
        self.final_time = final_time
        self.step_size = step_size

        # Initialize history arrays. During propagation there will be N timesteps plus the initial timestep so the
        # arrays need to be of size N+1.
        timesteps = int(np.floor((self.final_time - orbit.time) / self.step_size))
        self.position_history = np.zeros([3, timesteps + 1])
        self.velocity_history = np.zeros([3, timesteps + 1])
        self.time_history = np.zeros([1, timesteps + 1])

        # Assign initial conditions to the history arrays.
        self.position_history[:, 0] = orbit.position
        self.velocity_history[:, 0] = orbit.velocity
        self.time_history[0, 0] = orbit.time

    @abstractmethod
    def propagate(self):
        pass


class Keplerian(Propagator):
    """
    Propagator which uses Kepler's equation along with f and g series.

    :ivar fg_constraint: Whether to compute the gdot-series independently (increasing computation time) or to instead
        use the series constraint (faster but less accurate).
    :ivar tol: Tolerance to use when solving Kepler's equation.
    """

    def __init__(self, orbit, final_time, step_size, tol, fg_constraint=True):
        self.fg_constraint = fg_constraint
        self.tol = tol
        super().__init__(orbit, final_time, step_size)

    def propagate(self):
        """
        The procedure for this style of propagation is as follows:
            1) Save initial position and velocity as well as the initial eccentric anomaly.
            2) Compute the new eccentric anomaly on the next time step from Kepler's equation.
            3) Form the f and g functions and use them to compute the new position.
            4) Form the fdot and gdot functions and use them and the new position to compute the new velocity.
            5) Repeat 2-4 until the final time is reached.
        """

        # Get initial values used for propagation.
        initial_time = self.orbit.time
        initial_position = self.orbit.position.copy()
        initial_velocity = self.orbit.velocity.copy()
        initial_eccentric_anomaly = self.gauss_equation()

        for timestep in range(1, self.time_history.shape[1]):
            # Compute new eccentric anomaly.
            self.orbit.time += self.step_size
            eccentric_anomaly = self.kepler_equation(initial_eccentric_anomaly, initial_time)

            # Compute the f and g functions.
            f_func = (
                    1 - self.orbit.sm_axis / np.linalg.norm(initial_position)
                        * (1 - np.cos(eccentric_anomaly - initial_eccentric_anomaly))
            )
            g_func = (
                    (self.orbit.time - initial_time) - 1 / np.sqrt(self.orbit.grav_param / self.orbit.sm_axis ** 3)
                        * (eccentric_anomaly - initial_eccentric_anomaly
                            - np.sin(eccentric_anomaly - initial_eccentric_anomaly))
            )

            # Compute new position.
            self.orbit.position = f_func * initial_position + g_func * initial_velocity

            # Compute fdot and gdot functions.
            fdot_func = (
                -np.sqrt(self.orbit.grav_param * self.orbit.sm_axis)
                    / (np.linalg.norm(initial_position) * np.linalg.norm(self.orbit.position))
                    * np.sin(eccentric_anomaly - initial_eccentric_anomaly)
            )
            if self.fg_constraint:  # Only compute gdot function manually if constraint usage is disabled.
                gdot_func = (g_func * fdot_func + 1) / f_func
            else:
                gdot_func = (
                        1 - self.orbit.sm_axis / np.linalg.norm(self.orbit.position)
                            * (1 - np.cos(eccentric_anomaly - initial_eccentric_anomaly))
                )

            # Compute new velocities.
            self.orbit.velocity = fdot_func * initial_position + gdot_func * initial_velocity

            # Add results to history arrays.
            self.time_history[0, timestep] = self.orbit.time
            self.position_history[:, timestep] = self.orbit.position
            self.velocity_history[:, timestep] = self.orbit.velocity

    def gauss_equation(self):
        """
        Function used to convert true anomaly to eccentric anomaly.

        :return: Eccentric anomaly.
        """

        return (
                2 * np.arctan(np.sqrt((1 - self.orbit.eccentricity) / (1 + self.orbit.eccentricity))
                    * np.tan(self.orbit.true_anomaly / 2))
        )

    def kepler_equation(
            self,
            initial_eccentric_anomaly,
            initial_time,
    ):
        """
        Function used to compute the new eccentric anomaly given the current eccentric anomaly and the desired time
        increment. Kepler's equation is transcendental wrt. eccentric anomaly so root-finding via sp.optimize.newton()
        is used to solve for it.

        :return: New eccentric anomaly at the current time plus the desired timestep.
        """

        # Root-finding.
        eq = lambda x: (
                np.sqrt(self.orbit.grav_param / self.orbit.sm_axis ** 3) * (self.orbit.time - initial_time)
                    + initial_eccentric_anomaly - self.orbit.eccentricity * np.sin(initial_eccentric_anomaly)
                    -  x + self.orbit.eccentricity * np.sin(x)
        )
        eccentric_anomaly = sp.optimize.newton(eq, initial_eccentric_anomaly, tol=self.tol)

        return eccentric_anomaly
