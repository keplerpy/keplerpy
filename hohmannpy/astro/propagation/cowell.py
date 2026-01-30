from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np
import scipy as sp

from . import base

if TYPE_CHECKING:
    from .. import mission, perturbations


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
            absolute_solver_tol: float = 1e-15,
            relative_solver_tol: float = 1e-12,
    ):
        self.absolute_solver_tol = absolute_solver_tol
        self.relative_solver_tol = relative_solver_tol

        super().__init__(step_size)

    def propagate(
            self,
            satellites: dict[str, mission.Satellite],
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
        initial_times = {}
        initial_positions = {}
        initial_velocities = {}

        for name, satellite in self.satellites.items():
            initial_times[name] = satellite.orbit.time
            initial_positions[name] = satellite.orbit.position.copy()
            initial_velocities[name] = satellite.orbit.velocity.copy()

            for logger in satellite.loggers:
                logger.setup(initial_orbit=satellite.orbit, timesteps=self.timesteps)

        # Use scipy's solve_ivp() to numerically integrate the equations of motion.
        for name, satellite in self.satellites.items():
            orbit = satellite.orbit

            initial_state = np.hstack((initial_positions[name], initial_velocities[name]))
            eval_times = np.linspace(initial_times[name], final_time, self.timesteps + 1)
            sol = sp.integrate.solve_ivp(
                self.eom,
                [initial_times[name], final_time],
                initial_state,
                t_eval=eval_times,
                atol=self.absolute_solver_tol,
                rtol=self.relative_solver_tol,
                args=[orbit.grav_param]
            )

            # Extract propagation results.
            for timestep in range(1, self.timesteps + 1):
                orbit.time += self.step_size

                # Extract position.
                orbit.position = sol.y[0:3, timestep]

                # Extract velocity.
                orbit.velocity = sol.y[3:, timestep]

                # Update orbital elements.
                orbit.update_classical()
                if orbit.track_equinoctial:
                    orbit.update_equinoctial()

                # Save results.
                self.log(timestep)

    def eom(self, t, y, grav_param):
        """
        Equation of motion passed to solve_ivp(). This is simply the two-body equation of motion (optionally with added
        perturbing forces) put in first-order form.

        :param t: Current time.
        :param y: Current state, a (6, ) numpy array where indices 0-2 correspond to the (x, y, z) Cartesian position
            and indices 4-5 the accompanying (x, y, z) Cartesian velocity.
        """

        # Standard two-body acceleration.
        y0_dot = y[3]
        y1_dot = y[4]
        y2_dot = y[5]

        radius = np.sqrt(y[0] ** 2 + y[1] ** 2 + y[2] ** 2)

        y3_dot = -grav_param / radius ** 3 * y[0]
        y4_dot = -grav_param / radius ** 3 * y[1]
        y5_dot = -grav_param / radius ** 3 * y[2]

        # Perturbing forces.
        if self.perturbing_forces is not None:
            for perturbing_force in self.perturbing_forces:
                y3_perturb, y4_perturb, y5_perturb = perturbing_force.evaluate(t, y)
                y3_dot += y3_perturb
                y4_dot += y4_perturb
                y5_dot += y5_perturb

        return y0_dot, y1_dot, y2_dot, y3_dot, y4_dot, y5_dot
