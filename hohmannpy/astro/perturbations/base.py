from __future__ import annotations
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .. import satellites


class Perturbation(ABC):
    r"""
    Base class for implementing perturbations from Keplerian two-body orbital mechanics. These are designed to be used
    in conjunction with a non-Keplerian propagator such as :class:`~hohmannpy.astro.CowellPropagator` or
    :class:`~hohmannpy.astro.EnckePropagator`.

    Child classes must implement :meth:`evaluate()`. This is called on each timestep of propagation by
    :class:`~hohmannpy.astro.Propagator` . :meth:`~hohmannpy.astro.Propagator.propagate()` to return thw perturbing
    acceleration for a given orbital state.
    """

    def __init__(self):
        pass

    @abstractmethod
    def evaluate(self, time: float, state: np.ndarray, satellite: satellites.Satellite) -> np.ndarray:
        r"""
        Takes in the current time and planet-centered inertial state (the position and velocity) and returns the
        perturbing acceleration.

        Parameters
        ----------
        time : float
            Current time in seconds since propagation began.
        state : np.ndarray
            Current translational state in planet-centered inertial coordinates given as (position, velocity).
        """

        pass
