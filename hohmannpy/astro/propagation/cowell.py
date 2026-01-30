from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np

from . import base

if TYPE_CHECKING:
    from .. import spacecraft, perturbations


class CowellPropagator(base.Propagator):
    """
    Propagator which simply numerically integrates the equations of motion of the orbit using scipy's solve_ivp(). By
    default, only the gravity of the central body is considered but unlike Keplerian methods perturbing forces may also
    be included.
        By default, RK45 (a Runge-Kutta method of the 5th-order with a 4th-order error estimate) is used for
    numerical integration.

    NOTE: Numerical integration's accuracy is wholly based on the solver tolerances unlike Keplerian methods. This is
    because integration error accumulates whereas the root-finding error used to implement analytic solution is
    constant. As a result much stricter tolerances are needed for the solver to maintain accuracy.
    """

    def __init__(
            self,
            step_size: float = None,
    ):
        super().__init__(step_size)

    def propagate(
            self,
            satellites: dict[str, spacecraft.Satellite],
            final_time: float,
            perturbing_forces: list[perturbations.Perturbation] = None
    ):
        """
        The procedure for this style of propagation is as follows:
            1) Save initial position and velocity.
            2) Call solve_ivp() and integrate the equations of motion numerically.
            3) Extract the results of this function and save them.
        """

        super().propagate(satellites, final_time, perturbing_forces)

        # Get initial values used for propagation and set up logging capabilities.

        for name, satellite in self.satellites.items():
            for logger in satellite.loggers:
                logger.setup(initial_orbit=satellite.orbit, timesteps=self.timesteps)

        # Begin the actual propagation loop
        for timestep in range(1, self.timesteps + 1):
            for name, satellite in self.satellites.items():
                orbit = satellite.orbit
                state = self.rk4(
                    t0=orbit.time,
                    y0=np.concatenate((orbit.position, orbit.velocity)),
                    satellite=satellite,
                )
                orbit.time += self.step_size
                orbit.position = np.array(state[:3])
                orbit.velocity = np.array(state[3:])

                orbit.update_classical()
                if orbit.track_equinoctial:
                    orbit.update_equinoctial()

            # Save results from this timestep.
            self.log(timestep)

    def eom(
            self,
            t: float,
            y: np.ndarray,
            satellite: spacecraft.Satellite
    ) -> np.ndarray:
        radius = np.sqrt(y[0] ** 2 + y[1] ** 2 + y[2] ** 2)

        y0_dot = y[3]
        y1_dot = y[4]
        y2_dot = y[5]

        y3_dot = -satellite.orbit.grav_param / radius ** 3 * y[0]
        y4_dot = -satellite.orbit.grav_param / radius ** 3 * y[1]
        y5_dot = -satellite.orbit.grav_param / radius ** 3 * y[2]

        # Perturbing forces.
        if self.perturbing_forces is not None:
            for perturbing_force in self.perturbing_forces:
                y3_perturb, y4_perturb, y5_perturb = perturbing_force.evaluate(t, y)
                y3_dot += y3_perturb
                y4_dot += y4_perturb
                y5_dot += y5_perturb

        return np.array([y0_dot, y1_dot, y2_dot, y3_dot, y4_dot, y5_dot])

    def rk4(
            self,
            t0: float,
            y0: np.ndarray,
            satellite: spacecraft.Satellite
    ) -> np.ndarray:
        x1 = self.eom(t0, y0, satellite)
        x2 = self.eom(t0 + self.step_size / 2, y0 + self.step_size / 2 * x1, satellite)
        x3 = self.eom(t0 + self.step_size / 2, y0 + self.step_size / 2 * x2, satellite)
        x4 = self.eom(t0 + self.step_size, y0 + self.step_size * x3, satellite)

        return y0 + self.step_size / 6 * (x1 + 2 * x2 + 2 * x3 + x4)
