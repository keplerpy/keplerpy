from __future__ import annotations
from .. import logging
from . import base
import numpy as np
import scipy as sp


class KeplerPropagator(base.Propagator):
    """
    Propagator which uses Kepler's equation along with f and g series. If eccentricity is greater than 1 automatically
    switches over to using the hyperbolic eccentric anomaly. The parabolic case is not included.

    :ivar fg_constraint: Whether to compute the gdot-series independently (increasing computation time) or to instead
        use the series constraint (faster but less accurate).
    :ivar solver_tol: Tolerance to use when solving Kepler's equation.

    [PROPAGATION METHOD PARAMETERS]
    :ivar eccentric_anomaly:
    """

    def __init__(
            self,
            loggers: list[logging.Logger] = None,
            step_size: float = None,
            solver_tol: float = 1e-8,
            fg_constraint: bool = True
    ):
        self.fg_constraint = fg_constraint
        self.solver_tol = solver_tol

        self.eccentric_anomaly = None

        if loggers is None:  # Default loggers.
            loggers = [logging.StateLogger(), logging.EccentricAnomalyLogger()]

        super().__init__(loggers, step_size)

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
        self.eccentric_anomaly = initial_eccentric_anomaly

        # Set up Loggers.
        for logger in self.loggers:
            logger.setup(self)

        for timestep in range(1, self.timesteps + 1):
            self.orbit.time += self.step_size

            # -------------
            # ELLIPTIC CASE
            # -------------
            if self.orbit.eccentricity < 1:  # Elliptical case.
                # Compute new eccentric anomaly. Use the previous eccentric anomaly as the initial guess for the
                # root-finder.
                self.eccentric_anomaly = self.kepler_equation(
                    initial_eccentric_anomaly=initial_eccentric_anomaly,
                    initial_guess=self.eccentric_anomaly,
                    initial_time=initial_time
                )

                # Compute the f and g functions.
                f_func = (
                        1 - self.orbit.sm_axis / np.linalg.norm(initial_position)
                            * (1 - np.cos(self.eccentric_anomaly - initial_eccentric_anomaly))
                )
                g_func = (
                        self.orbit.time - initial_time - 1 / np.sqrt(self.orbit.grav_param / self.orbit.sm_axis ** 3)
                            * (self.eccentric_anomaly - initial_eccentric_anomaly
                                - np.sin(self.eccentric_anomaly - initial_eccentric_anomaly))
                )

                # Compute new position (and true anomaly).
                self.orbit.position = f_func * initial_position + g_func * initial_velocity
                self.orbit.update_true_anomaly()
                self.orbit.update_argl()
                self.orbit.update_true_latitude()

                # Compute fdot and gdot functions.
                fdot_func = (
                    -np.sqrt(self.orbit.grav_param * self.orbit.sm_axis)
                        / (np.linalg.norm(initial_position) * np.linalg.norm(self.orbit.position))
                        * np.sin(self.eccentric_anomaly - initial_eccentric_anomaly)
                )
                if self.fg_constraint:  # Only compute gdot function manually if constraint usage is disabled.
                    gdot_func = (g_func * fdot_func + 1) / f_func
                else:
                    gdot_func = (
                            1 - self.orbit.sm_axis / np.linalg.norm(self.orbit.position)
                                * (1 - np.cos(self.eccentric_anomaly - initial_eccentric_anomaly))
                    )

            # ---------------
            # HYPERBOLIC CASE
            # ---------------
            else:
                # Compute new eccentric anomaly. Use the previous eccentric anomaly as the initial guess for the
                # root-finder.
                self.eccentric_anomaly = self.kepler_equation(
                    initial_eccentric_anomaly=initial_eccentric_anomaly,
                    initial_guess=self.eccentric_anomaly,
                    initial_time=initial_time
                )

                # Compute f and g functions.
                f_func = (
                        1 - self.orbit.sm_axis / np.linalg.norm(initial_position)
                            * (1 - np.cosh(self.eccentric_anomaly - initial_eccentric_anomaly))
                )
                g_func = (
                        self.orbit.time - initial_time
                            - 1 / np.sqrt(self.orbit.grav_param / (-self.orbit.sm_axis) ** 3)
                            * (np.sinh(self.eccentric_anomaly - initial_eccentric_anomaly)
                                - (self.eccentric_anomaly - initial_eccentric_anomaly))
                )

                # Compute new position (and true anomaly).
                self.orbit.position = f_func * initial_position + g_func * initial_velocity
                self.orbit.update_true_anomaly()
                self.orbit.update_argl()
                self.orbit.update_true_latitude()

                # Compute fdot and gdot functions.
                fdot_func = (
                        -np.sqrt(self.orbit.grav_param * -self.orbit.sm_axis)
                        / (np.linalg.norm(initial_position) * np.linalg.norm(self.orbit.position))
                        * np.sinh(self.eccentric_anomaly - initial_eccentric_anomaly)
                )
                if self.fg_constraint:  # Only compute gdot function manually if constraint usage is disabled.
                    gdot_func = (g_func * fdot_func + 1) / f_func
                else:
                    gdot_func = (
                            1 - self.orbit.sm_axis / np.linalg.norm(self.orbit.position)
                            * (1 - np.cosh(self.eccentric_anomaly - initial_eccentric_anomaly))
                    )

            # Compute new velocities.
            self.orbit.velocity = fdot_func * initial_position + gdot_func * initial_velocity

            # Save results from this timestep.
            self.log(timestep)

    def gauss_equation(self):
        """
        Function used to convert true anomaly to eccentric anomaly.

        :return: Eccentric anomaly.
        """

        if self.orbit.eccentricity < 1:  # Elliptic case.
            return (
                    2 * np.arctan(np.sqrt((1 - self.orbit.eccentricity) / (1 + self.orbit.eccentricity))
                        * np.tan(self.orbit.true_anomaly / 2))
            )
        else:  # Hyperbolic case.
            return (
                    2 * np.arctanh(np.sqrt((self.orbit.eccentricity - 1) / (self.orbit.eccentricity + 1))
                                  * np.tan(self.orbit.true_anomaly / 2))
            )

    def kepler_equation(
            self,
            initial_eccentric_anomaly: float,
            initial_guess: float,
            initial_time: float,
    ) -> float:
        """
        Function used to compute the new eccentric anomaly given the current eccentric anomaly and the desired time
        increment. Kepler's equation is transcendental wrt. eccentric anomaly so root-finding via sp.optimize.newton()
        is used to solve for it. The ideal initial guess is just the eccentric anomaly on the previous timestep.

        :param initial_eccentric_anomaly: Eccentric anomaly at epoch.
        :param initial_guess: Eccentric anomaly from the last iteration.
        :param initial_time: Time at epoch.

        :return: New eccentric anomaly at the current time plus the desired timestep.
        """

        # Root-finding.
        if self.orbit.eccentricity < 1:  # Elliptic case.
            eq = lambda x: (
                    np.sqrt(self.orbit.grav_param / self.orbit.sm_axis ** 3) * (self.orbit.time - initial_time)
                        + initial_eccentric_anomaly - self.orbit.eccentricity * np.sin(initial_eccentric_anomaly)
                        -  x + self.orbit.eccentricity * np.sin(x)
            )
        else:  # Hyperbolic case.
            eq = lambda x: (
                    np.sqrt(self.orbit.grav_param / (-self.orbit.sm_axis) ** 3) * (self.orbit.time - initial_time)
                    + self.orbit.eccentricity * np.sinh(initial_eccentric_anomaly) - initial_eccentric_anomaly
                    - self.orbit.eccentricity * np.sinh(x) + x
            )
        eccentric_anomaly = sp.optimize.newton(eq, initial_guess, tol=self.solver_tol)

        return eccentric_anomaly
