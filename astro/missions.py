from . import propagation, orbits
from ui import rendering
import numpy as np
import pandas as pd


class Mission:
    def __init__(
            self,
            starting_orbit: orbits.Orbit,
            maneuvers,
            initial_global_time: float,
            final_global_time: float,
            propagator: propagation.base.Propagator = None,
            satellite=None,
    ):
        # Instantiate all the passed-in attributes.
        self.starting_orbit = starting_orbit
        self.global_time = initial_global_time
        self.initial_global_time = initial_global_time
        self.final_global_time = final_global_time

        # For both the propagator and satellite a default option exists if the user does not input one, if they did
        # ignore and simply instantiate as normal.
        if satellite is None:
            self.satellite = ...  # TODO: Add a generic cube-sat.
        else:
            self.satellite = satellite

        if propagator is None:
            self.propagator = propagation.universal_variable.UniversalVariablePropagator(solver_tol=1e-8)
        else:
            self.propagator = propagator

        # TODO: Add maneuvers.
        self.maneuvers = maneuvers

        # Pre-instantiate attributes to be assigned later.
        self.traj = ...

    def simulate(self):
        """
        Call the propagator's propagate() function to generate the orbital trajectory and then log it.
        """

        self.propagator.setup(
            orbit=self.starting_orbit,
            final_time=self.final_global_time
        )
        self.propagator.propagate()

        # TODO: Error handling for missing a state logger.
        for logger in self.propagator.loggers:
            if isinstance(logger, propagation.logging.StateLogger):
                self.traj = logger.position_history
                break

    def display(self):
        """
        Use pygfx to display the resulting trajectory.
        """

        # TODO: Add an error here if simulate() has not yet been called.
        engine = rendering.TempRenderEngine()
        engine.draw_orbit(self.traj)
        engine.render()

    def to_csv(self, file_path, fp_accuracy=6):
        """
        Save the resulting trajectory to a .csv file.

        :param file_path: Name and destination of resultant .csv file.
        :param fp_accuracy: How many sig figs past the decimal data should be logged to.
        """

        # TODO: Error handling for missing a state logger.
        for logger in self.propagator.loggers:
            if isinstance(logger, propagation.logging.StateLogger):
                times_to_log = logger.time_history.copy()
                positions_to_log = logger.position_history.copy()
                velocities_to_log = logger.velocity_history.copy()

                times_to_log = times_to_log.T
                positions_to_log = positions_to_log.T
                velocities_to_log = velocities_to_log.T

                labels = ['time', 'x-position', 'y-position', 'z-position', 'x-velocity', 'y-velocity', 'z-velocity']
                data_arr = np.hstack((times_to_log, positions_to_log, velocities_to_log))
                data_df = pd.DataFrame(data_arr, columns=labels)
                data_df.to_csv(
                    f"{file_path}.csv",
                    index=False,
                    float_format=f"%.{fp_accuracy}f"
                )
                break
