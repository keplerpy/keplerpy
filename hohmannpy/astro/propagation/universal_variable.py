from __future__ import annotations
from . import base
import numpy as np
from numpy.typing import NDArray
import scipy as sp
from .. import logging


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
            loggers: list[logging.Logger] = None,
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

        if loggers is None:  # Default loggers.
            loggers = [logging.StateLogger(), logging.UniversalVariableLogger()]

        super().__init__(loggers, step_size)

    def propagate(self):
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

        # Get initial values used for propagation.
        initial_time = self.orbit.time
        initial_position = self.orbit.position.copy()
        initial_velocity = self.orbit.velocity.copy()
        self.universal_variable = 0  # By definition always starts at 0 when propagation begins.

        # Set up Loggers.
        for logger in self.loggers:
            logger.setup(self)

        # Compute the inverse of the semi-major axis. This is needed to handle parabolic orbits where otherwise division
        # by a semi-major axis of 0 would occur.
        self.inverse_sm_axis = (
            (2 * self.orbit.grav_param / np.linalg.norm(initial_position) - np.linalg.norm(initial_velocity) ** 2)
                / self.orbit.grav_param
        )

        # Propagation.
        for timestep in range(1, self.timesteps + 1):
            self.orbit.time += self.step_size

            # Compute new universal variable. Use the previous universal variable as the initial guess for the
            # root-finder.
            self.universal_variable = self.kepler_equation(
                initial_time=initial_time,
                initial_position=initial_position,
                initial_velocity=initial_velocity,
                initial_guess=self.universal_variable,
            )

            # Compute the Stumpff (c and s) functions.
            self.stumpff_param = self.inverse_sm_axis * self.universal_variable ** 2
            s_func, c_func = self.stumpff_funcs(self.stumpff_param)

            # Compute the f and g functions.
            f_func = 1 - self.universal_variable ** 2 / np.linalg.norm(initial_position) * c_func
            g_func = (
                    self.orbit.time - initial_time
                        - self.universal_variable ** 3 / np.sqrt(self.orbit.grav_param) * s_func
            )

            # Compute new position (and true anomaly).
            self.orbit.position = f_func * initial_position + g_func * initial_velocity
            self.orbit.update_true_anomaly()
            self.orbit.update_argl()
            self.orbit.update_true_latitude()

            # Compute fdot and gdot functions.
            fdot_func = (
                    np.sqrt(self.orbit.grav_param)
                        / (np.linalg.norm(self.orbit.position) * np.linalg.norm(initial_position))
                        * self.universal_variable * (self.stumpff_param * s_func - 1)
            )
            if self.fg_constraint:  # Only compute gdot function manually if constraint usage is disabled.
                gdot_func = (g_func * fdot_func + 1) / f_func
            else:
                gdot_func = 1 - self.universal_variable ** 2 / np.linalg.norm(self.orbit.position) * c_func

            # Compute new velocities.
            self.orbit.velocity = fdot_func * initial_position + gdot_func * initial_velocity

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
            initial_time: float,
            initial_position: NDArray[float],
            initial_velocity: NDArray[float],
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
            stumpff_param = self.inverse_sm_axis * x ** 2
            s_func, c_func = self.stumpff_funcs(stumpff_param)

            return (
                    x ** 3 * s_func
                        + np.dot(initial_position, initial_velocity) / np.sqrt(self.orbit.grav_param)
                        * x ** 2 * c_func
                        + np.linalg.norm(initial_position) * x * (1 - stumpff_param * s_func)
                        - np.sqrt(self.orbit.grav_param) * (self.orbit.time - initial_time)
            )

        # Root-finding.
        universal_variable = sp.optimize.newton(eq, initial_guess, tol=self.solver_tol)

        return universal_variable
