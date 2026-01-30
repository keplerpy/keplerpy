from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np
import scipy as sp

from . import base

if TYPE_CHECKING:
    from .. import spacecraft, perturbations


class KeplerPropagator(base.Propagator):
    """
    Propagator which uses Kepler's equation along with f and g series. If eccentricity is greater than 1 automatically
    switches over to using the hyperbolic eccentric anomaly. The parabolic case is not included.
    """

    def __init__(
            self,
            step_size: float = None,
            solver_tol: float = 1e-8,
            fg_constraint: bool = True
    ):
        self.fg_constraint = fg_constraint
        self.solver_tol = solver_tol

        super().__init__(step_size)

    def propagate(
            self,
            satellites: dict[str, spacecraft.Satellite],
            final_time: float,
            perturbing_forces: list[perturbations.Perturbation] = None
    ):
        """
        The procedure for this style of propagation is as follows:
            1) Save initial position and velocity as well as the initial eccentric anomaly.
            2) Compute the new eccentric anomaly on the next time step from Kepler's equation.
            3) Form the f and g functions and use them to compute the new position.
            4) Form the fdot and gdot functions and use them and the new position to compute the new velocity.
            5) Repeat 2-4 until the final time is reached.
        """

        super().propagate(satellites, final_time, perturbing_forces)

        # Get initial values used for propagation and set up logging capabilities.
        initial_times = {}
        initial_positions = {}
        initial_velocities = {}
        initial_eccentric_anomalies = {}

        for name, satellite in self.satellites.items():
            initial_times[name] = satellite.orbit.time
            initial_positions[name] = satellite.orbit.position.copy()
            initial_velocities[name] = satellite.orbit.velocity.copy()
            initial_eccentric_anomalies[name] = (
                self.gauss_equation(
                    eccentricity=satellite.orbit.eccentricity,
                    true_anomaly=satellite.orbit.true_anomaly
                )
            )

            satellite.orbit.eccentric_anomaly = initial_eccentric_anomalies[name]  # Needed for logging purposes.

            for logger in satellite.loggers:
                logger.setup(initial_orbit=satellite.orbit, timesteps=self.timesteps)

        # Begin the actual propagation loop
        for timestep in range(1, self.timesteps + 1):
            for name, satellite in self.satellites.items():
                orbit = satellite.orbit
                orbit.time += self.step_size

                # -------------
                # ELLIPTIC CASE
                # -------------
                if orbit.eccentricity < 1:  # Elliptical case.
                    # Compute new eccentric anomaly. Use the previous eccentric anomaly as the initial guess for the
                    # root-finder.
                    orbit.eccentric_anomaly = self.kepler_equation(
                        time=orbit.time,
                        eccentricity=orbit.eccentricity,
                        sm_axis=orbit.sm_axis,
                        grav_param=orbit.grav_param,
                        initial_eccentric_anomaly=initial_eccentric_anomalies[name],
                        initial_guess=orbit.eccentric_anomaly,
                        initial_time=initial_times[name]
                    )

                    # Compute the f and g functions.
                    f_func = (
                            1 - orbit.sm_axis / np.linalg.norm(initial_positions[name])
                                * (1 - np.cos(orbit.eccentric_anomaly - initial_eccentric_anomalies[name]))
                    )
                    g_func = (
                            orbit.time - initial_times[name]
                                - 1 / np.sqrt(orbit.grav_param / orbit.sm_axis ** 3)
                                * (orbit.eccentric_anomaly - initial_eccentric_anomalies[name]
                                    - np.sin(orbit.eccentric_anomaly - initial_eccentric_anomalies[name]))
                    )

                    # Compute new position (and true anomaly).
                    orbit.position = (
                            f_func * initial_positions[name] + g_func * initial_velocities[name]
                    )
                    orbit.update_true_anomaly()
                    orbit.update_argl()
                    orbit.update_true_latitude()

                    # Compute fdot and gdot functions.
                    fdot_func = (
                        -np.sqrt(orbit.grav_param * orbit.sm_axis)
                            / (np.linalg.norm(initial_positions[name]) * np.linalg.norm(orbit.position))
                            * np.sin(orbit.eccentric_anomaly - initial_eccentric_anomalies[name])
                    )
                    if self.fg_constraint:  # Only compute gdot function manually if constraint usage is disabled.
                        gdot_func = (g_func * fdot_func + 1) / f_func
                    else:
                        gdot_func = (
                                1 - orbit.sm_axis / np.linalg.norm(orbit.position)
                                    * (1 - np.cos(orbit.eccentric_anomaly - initial_eccentric_anomalies[name]))
                        )

                # ---------------
                # HYPERBOLIC CASE
                # ---------------
                else:
                    # Compute new eccentric anomaly. Use the previous eccentric anomaly as the initial guess for the
                    # root-finder.
                    orbit.eccentric_anomaly = self.kepler_equation(
                        time=orbit.time,
                        eccentricity=orbit.eccentricity,
                        sm_axis=orbit.sm_axis,
                        grav_param=orbit.grav_param,
                        initial_eccentric_anomaly=initial_eccentric_anomalies[name],
                        initial_guess=orbit.eccentric_anomaly,
                        initial_time=initial_times[name]
                    )

                    # Compute f and g functions.
                    f_func = (
                            1 - orbit.sm_axis / np.linalg.norm(initial_positions[name])
                                * (1 - np.cosh(orbit.eccentric_anomaly - initial_eccentric_anomalies[name]))
                    )
                    g_func = (
                            orbit.time - initial_times[name]
                                - 1 / np.sqrt(orbit.grav_param / (-orbit.sm_axis) ** 3)
                                * (np.sinh(orbit.eccentric_anomaly - initial_eccentric_anomalies[name])
                                    - (orbit.eccentric_anomaly - initial_eccentric_anomalies[name]))
                    )

                    # Compute new position (and true anomaly).
                    orbit.position = (
                            f_func * initial_positions[name] + g_func * initial_velocities[name]
                    )
                    orbit.update_true_anomaly()
                    orbit.update_argl()
                    orbit.update_true_latitude()

                    # Compute fdot and gdot functions.
                    fdot_func = (
                            -np.sqrt(orbit.grav_param * -orbit.sm_axis)
                            / (np.linalg.norm(initial_positions[name]) * np.linalg.norm(orbit.position))
                            * np.sinh(orbit.eccentric_anomaly - initial_eccentric_anomalies[name])
                    )
                    if self.fg_constraint:  # Only compute gdot function manually if constraint usage is disabled.
                        gdot_func = (g_func * fdot_func + 1) / f_func
                    else:
                        gdot_func = (
                                1 - orbit.sm_axis / np.linalg.norm(orbit.position)
                                * (1 - np.cosh(orbit.eccentric_anomaly - initial_eccentric_anomalies[name]))
                        )

                # Compute new velocities.
                orbit.velocity = (
                        fdot_func * initial_positions[name] + gdot_func * initial_velocities[name]
                )

            # Save results from this timestep.
            self.log(timestep)

    def gauss_equation(self, eccentricity, true_anomaly):
        """
        Function used to convert true anomaly to eccentric anomaly.
        """

        if eccentricity < 1:  # Elliptic case.
            return (
                    2 * np.arctan(np.sqrt((1 - eccentricity) / (1 + eccentricity))
                        * np.tan(true_anomaly / 2))
            )
        else:  # Hyperbolic case.
            return (
                    2 * np.arctanh(np.sqrt((eccentricity - 1) / (eccentricity + 1))
                                  * np.tan(true_anomaly / 2))
            )

    def kepler_equation(
            self,
            time: float,
            eccentricity: float,
            sm_axis: float,
            grav_param: float,
            initial_eccentric_anomaly: float,
            initial_guess: float,
            initial_time: float,
    ) -> float:
        """
        Function used to compute the new eccentric anomaly given the current eccentric anomaly and the desired time
        increment. Kepler's equation is transcendental wrt. eccentric anomaly so root-finding via sp.optimize.newton()
        is used to solve for it. The ideal initial guess is just the eccentric anomaly on the previous timestep.a
        """

        # Root-finding.
        if eccentricity < 1:  # Elliptic case.
            eq = lambda x: (
                    np.sqrt(grav_param / sm_axis ** 3) * (time - initial_time)
                        + initial_eccentric_anomaly - eccentricity * np.sin(initial_eccentric_anomaly)
                        - x + eccentricity * np.sin(x)
            )
        else:  # Hyperbolic case.
            eq = lambda x: (
                    np.sqrt(grav_param / (-sm_axis) ** 3) * (time - initial_time)
                    + eccentricity * np.sinh(initial_eccentric_anomaly) - initial_eccentric_anomaly
                    - eccentricity * np.sinh(x) + x
            )
        eccentric_anomaly = sp.optimize.newton(eq, initial_guess, tol=self.solver_tol)

        return eccentric_anomaly
