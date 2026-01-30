from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .. import perturbations, mission


class Propagator:
    def __init__(self, step_size: float = None):
        self.step_size = step_size

        self.satellites = None
        self.perturbing_forces = None
        self.timesteps = None

    def propagate(
            self,
            satellites: dict[str, mission.Satellite],
            final_time: float,
            perturbing_forces: list[perturbations.Perturbation] = None
    ):
        self.satellites = satellites
        self.perturbing_forces = perturbing_forces

        # Compute number of timesteps to propagate for and use this information to set up the Loggers.
        if self.step_size is None:  # Default to once a minute.
            self.step_size = 60
        self.timesteps = int(np.floor(final_time / self.step_size))

    def log(self, timestep):
        for satellite in self.satellites.values():
            for logger in satellite.loggers:
                logger.log(current_orbit=satellite.orbit, timestep=timestep)
