from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np
import scipy as sp

from . import base

if TYPE_CHECKING:
    from .. import spacecraft, perturbations


class UniversalVariablePropagator(base.Propagator):
    """
    Propagator which uses the universal variable formulation of Kepler's equation along with f and g series. This makes
    use of the titular "universal variable" along with special functions known as Stumpff series to handle propagation
    for the elliptical, hyperbolic, and parabolic cases seamlessly by switching between different definitions of said
    series.

    NOTE: In practice near-parabolic cases are handled equivalently to parabolic cases based on a tolerance set by the
    user. There is a loss in accuracy for these orbits based on how many terms in the Stumpff series are evaluated in
    these cases.

    :ivar fg_constraint: Whether to compute the gdot-series independently (increasing computation time) or to instead
        use the series constraint (faster but less accurate).
    :ivar solver_tol: Tolerance to use when solving Kepler's equation.
    :ivar stumpff_tol: Minimum absolute value of the Stumpff parameter before switching to the infinite series
        definition of the Stumpff series.
    :ivar stumpff_series_length: How many terms to evaluate in the Stumpff series when using their infinite series
        definitions.

    [PROPAGATION METHOD PARAMETERS]
    :ivar universal_variable:
    :ivar stumpff_param:
    """

    def __init__(
            self,
            step_size: float = None,
            solver_tol: float = 1e-8,
            stumpff_tol: float = 1e-8,
            stumpff_series_length: int = 10,
            fg_constraint: bool = True
    ):
        self.fg_constraint = fg_constraint
        self.solver_tol = solver_tol
        self.stumpff_tol = stumpff_tol
        self.stumpff_series_length = stumpff_series_length

        self.inverse_sm_axis = None
        self.universal_variable = None
        self.stumpff_param = None

        super().__init__(step_size)

    def propagate(
            self,
            satellites: dict[str, spacecraft.Satellite],
            final_time: float,
            perturbing_forces: list[perturbations.Perturbation] = None
    ):
        """
        The procedure for this style of propagation is as follows:
            1) Save initial position and velocity as well as the initial universal variable.
            2) Compute the new universal variable on the next time step from Kepler's equation. This involves computing
                the Stumpff series
            3) Recompute the Stumpff series using the new universal variable for use in computing the f and g functions.
            4) Form the f and g functions and use them to compute the new position.
            5) Form the fdot and gdot functions and use them and the new position to compute the new velocity.
            6) Repeat 2-5 until the final time is reached.
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

            satellite.orbit.universal_variable = 0  # Needed for logging purposes.
            satellite.orbit.stumpff_param = 0
            satellite.orbit.inverse_sm_axis = (
                (2 * satellite.orbit.grav_param / np.linalg.norm(initial_positions[name])
                    - np.linalg.norm(initial_velocities[name]) ** 2)
                        / satellite.orbit.grav_param
            )

            for logger in satellite.loggers:
                logger.setup(initial_orbit=satellite.orbit, timesteps=self.timesteps)

        # Propagation.
        for timestep in range(1, self.timesteps + 1):
            for name, satellite in self.satellites.items():
                orbit = satellite.orbit
                orbit.time += self.step_size

                # Compute new universal variable. Use the previous universal variable as the initial guess for the
                # root-finder.
                orbit.universal_variable = self.kepler_equation(
                    inverse_sm_axis=orbit.inverse_sm_axis,
                    grav_param=orbit.grav_param,
                    time=orbit.time,
                    initial_time=initial_times[name],
                    initial_position=initial_positions[name],
                    initial_velocity=initial_velocities[name],
                    initial_guess=orbit.universal_variable,
                )

                # Compute the Stumpff (c and s) functions.
                orbit.stumpff_param = orbit.inverse_sm_axis * orbit.universal_variable ** 2
                s_func, c_func = self.stumpff_funcs(orbit.stumpff_param)

                # Compute the f and g functions.
                f_func = 1 - orbit.universal_variable ** 2 / np.linalg.norm(initial_positions[name]) * c_func
                g_func = (
                        orbit.time - initial_times[name]
                            - orbit.universal_variable ** 3 / np.sqrt(orbit.grav_param) * s_func
                )

                # Compute new position (and true anomaly).
                orbit.position = f_func * initial_positions[name] + g_func * initial_velocities[name]
                orbit.update_true_anomaly()
                orbit.update_argl()
                orbit.update_true_latitude()

                # Compute fdot and gdot functions.
                fdot_func = (
                        np.sqrt(orbit.grav_param)
                            / (np.linalg.norm(orbit.position) * np.linalg.norm(initial_positions[name]))
                            * orbit.universal_variable * (orbit.stumpff_param * s_func - 1)
                )
                if self.fg_constraint:  # Only compute gdot function manually if constraint usage is disabled.
                    gdot_func = (g_func * fdot_func + 1) / f_func
                else:
                    gdot_func = 1 - orbit.universal_variable ** 2 / np.linalg.norm(orbit.position) * c_func

                # Compute new velocities.
                orbit.velocity = fdot_func * initial_positions[name] + gdot_func * initial_velocities[name]

            # Save results.
            self.log(timestep)

    def stumpff_funcs(self, stumpff_param) -> tuple[float, float]:
        """
        Special series which converge absolutely for any value of the Stumpff parameter, defined as the inverse of the
        semi-major axis times the universal variable squared. For "large" negative or positive values of the parameter
        these series have special closed-form hyperbolic or sinusoidal functions respectively. In the case where the
        parameter approaches zero instead the exact series definition must be used, in which case the number of terms is
        set by self.stumpff_series_length.

        :param stumpff_param: Stumpff parameter.

        :return: The "sine and cosine" Stumpff series referred to as the s_func and c_func.
        """

        if np.abs(stumpff_param) < self.stumpff_tol:  # Near-parabolic case.
            s_func = 0
            c_func = 0
            for i in range(self.stumpff_series_length):
                s_func += (-stumpff_param) ** i / sp.special.factorial(2 * i + 3)
                c_func += (-stumpff_param) ** i / sp.special.factorial(2 * i + 2)
        elif stumpff_param > 0:  # Elliptic case.
            s_func = (
                    (np.sqrt(stumpff_param) - np.sin(np.sqrt(stumpff_param))) / np.sqrt(stumpff_param ** 3)
            )
            c_func = (
                    (1 - np.cos(np.sqrt(stumpff_param))) / stumpff_param
            )
        else:  # Hyperbolic case.
            s_func = (
                    (np.sinh(np.sqrt(-stumpff_param)) - np.sqrt(-stumpff_param)) / np.sqrt(-stumpff_param ** 3)
            )
            c_func = (
                    (1 - np.cosh(np.sqrt(-stumpff_param))) / stumpff_param
            )

        return s_func, c_func

    def kepler_equation(
            self,
            inverse_sm_axis: float,
            grav_param: float,
            time: float,
            initial_time: float,
            initial_position: np.ndarray,
            initial_velocity: np.ndarray,
            initial_guess: float,
    ) -> float:
        """
        Function which yields the universal variable at the current time (as stored by self.orbit.time).

        :param initial_time: Time at start of propagation
        :param initial_position: Position at start of propagation, a (3, ) vector.
        :param initial_velocity: Velocity at start of propagation, a (3, ) vector.
        :param initial_guess: Initial guess for the universal variable.

        :return: New universal variable at the current time plus the desired timestep.
        """

        # Create the function to use in root-finding.
        def eq(x):
            stumpff_param = inverse_sm_axis * x ** 2
            s_func, c_func = self.stumpff_funcs(stumpff_param)

            return (
                    x ** 3 * s_func
                        + np.dot(initial_position, initial_velocity) / np.sqrt(grav_param)
                        * x ** 2 * c_func
                        + np.linalg.norm(initial_position) * x * (1 - stumpff_param * s_func)
                        - np.sqrt(grav_param) * (time - initial_time)
            )

        # Root-finding.
        universal_variable = sp.optimize.newton(eq, initial_guess, tol=self.solver_tol)

        return universal_variable
