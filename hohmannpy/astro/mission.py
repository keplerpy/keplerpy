from __future__ import annotations

import copy

import numpy as np
import pandas as pd

from . import propagation, orbit, perturbations, time, logging
from ..ui import rendering


class Mission:
    def __init__(
            self,
            satellites: list[Satellite],
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

    # TODO: Temp function.
    def display(self):
        if self.display_flag == "dynamic":
            engine = rendering.DynamicRenderEngine(
                traj=self.satellites["0"].position_history,
                times=self.satellites["0"].time_history,
                initial_global_time=self.initial_global_time,
            )
        else:
            engine = rendering.RenderEngine(
                traj=self.traj,
            )
        engine.render()


class Satellite:
    ballistic_coeff: float | None

    def __init__(
            self,
            name: str,
            starting_orbit: orbit.Orbit,
            mass: float = None,
            ballistic_coeff: float = None,
            mean_reflective_area: float = None,
            reflectivity: float = None
    ):
        self.name = name
        self.starting_orbit = starting_orbit
        self.mass = mass
        self.ballistic_coeff = ballistic_coeff
        self.mean_reflective_area = mean_reflective_area
        self.reflectivity = reflectivity

        self.orbit = copy.deepcopy(starting_orbit)  # This will be updated over time by the propagator.
        self.loggers = None  # Filled in by the __init__() of Mission.

    def __getattr__(self, name):
        if self.loggers is not None:
            for logger in self.loggers:
                if hasattr(logger, name):
                    return getattr(logger, name)
        raise AttributeError(f"This satellite has not logged data for {self.name}.")
