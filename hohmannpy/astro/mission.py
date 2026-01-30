from __future__ import annotations
import copy

import pandas as pd
import numpy as np

from . import propagation, perturbations, time, logging, spacecraft
from ..ui import rendering


class Mission:
    """
    Master class for all orbital simulations.

    Contains the ability to propagate the orbits of a set of :class:`~hohmannpy.astro.Satellite` and then render and
    propagate the results.

    Parameters
    ----------
    satellites : list of :class:`~hohmannpy.astro.Satellite`
        List of satellites whose orbits the Mission will propagate.
    initial_global_time : time.Time
        The Gregorian date and UT1 time to start the mission at.
    initial_global_time : time.Time
        The Gregorian date and UT1 time to end the mission at.
    loggers : list[:class:`~hohmannpy.astro.Logger`]
        Loggers determine which data to record for each satellite during propagation.
    """

    def __init__(
            self,
            satellites: list[spacecraft.Satellite],
            initial_global_time: time.Time,
            final_global_time: time.Time,
            loggers: list[logging.Logger] = None,
            propagator: propagation.base.Propagator = None,
            perturbing_forces: list[perturbations.Perturbation] = None,
            display: str = "dynamic"
    ):
        # Instantiate all the passed-in attributes.
        self.perturbing_forces = perturbing_forces
        self.display_flag = display
        self.initial_global_time = initial_global_time
        self.final_global_time = final_global_time
        self.global_time = initial_global_time

        # For both the propagator a default option exists if the user does not input one, if they did
        # ignore and simply instantiate as normal.
        if propagator is None:
            if perturbing_forces is None:
                self.propagator = propagation.universal_variable.UniversalVariablePropagator()
            else:
                self.propagator = propagation.cowell.CowellPropagator()
        else:
            self.propagator = propagator

        if loggers is None:
            loggers = [logging.StateLogger()]

        # Setup satellite data logging.
        self.satellites = {}
        for satellite in satellites:
            self.satellites[satellite.name] = satellite
            satellite.loggers = copy.deepcopy(loggers)

            # Raise error if satellite missing some attributes needed for an enabled perturbation.
            if self.perturbing_forces is not None:
                for perturbation in self.perturbing_forces:
                    if isinstance(perturbation, perturbations.AtmosphericDrag) and satellite.ballistic_coeff is None:
                        raise AttributeError("If AtmosphericDrag is enabled as a perturbation all satellites must have "
                                             "a value for the attribute 'ballistic coefficient'.")
                    if isinstance(perturbation, perturbations.SolarRadiation) and satellite.mass is None:
                        raise AttributeError("If SolarRadiation is enabled as a perturbation all satellites must have "
                                             "a value for the attribute 'mass'.")
                    if isinstance(perturbation, perturbations.SolarRadiation) and satellite.mean_reflective_area is None:
                        raise AttributeError("If SolarRadiation is enabled as a perturbation all satellites must have "
                                             "a value for the attribute 'mean reflective area'.")
                    if isinstance(perturbation, perturbations.SolarRadiation) and satellite.reflectivity is None:
                        raise AttributeError("If SolarRadiation is enabled as a perturbation all satellites must have"
                                             "a value for the attribute 'reflectivity'.")

    def simulate(self):
        self.propagator.propagate(
            satellites=self.satellites,
            perturbing_forces=self.perturbing_forces,
            final_time=(self.final_global_time.julian_date - self.initial_global_time.julian_date) * 86400,
        )

    def display(self):
        if self.display_flag == "dynamic":
            engine = rendering.DynamicRenderEngine(
                satellites=self.satellites,
                sim_length=(self.final_global_time.julian_date - self.initial_global_time.julian_date) * 86400,
                initial_global_time=self.initial_global_time,
            )
        else:
            engine = rendering.RenderEngine(
                satellites=self.satellites,
            )
        engine.render()

    def save(self, target_directory: str, fp_accuracy: float):
        for name, satellite in self.satellites.items():
            data = None
            labels = []

            for logger in satellite.loggers:
                local_data = logger.concatenate()
                local_labels = logger.labels

                if data is None:
                    data = local_data
                else:
                    data = np.hstack((data, local_data))
                labels.extend(local_labels)

            data_df = pd.DataFrame(data, columns=labels)
            data_df.to_csv(
                f"{target_directory}\\{name}.csv",
                index=False,
                float_format=f"%.{fp_accuracy}f"
            )
