from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np
from .. import orbit, perturbations, logging


class Propagator(ABC):
    """
    Base class for all propagators. All derivatives revolve around the method propagate() which takes in the initial
    orbital parameters (whatever those might be) and propagates them along the orbit up to the final time.
        This class effectively has two initializations. The first, the default __init__() is exposed to the user and is
    used to set solver settings. The second is the setup() method which fills in all the attributes needed to actually
    call propagate(). This is kept separate from __init__() because we want the former to only be for the user to set
    propagator settings before passing it into a Mission object. setup() can then be called internally by
    Mission.simulate() without exposing a bunch of extra code to the user.

    :ivar orbit: Orbit to perform propagation on. The position, and velocity, and time attributes of this object
        when it is passed in serve as the initial conditions of the orbit.
    :ivar perturbations: External forcing terms which cause deviations from the standard two-body equations of
        motion.
    :ivar final time: When to stop orbit propagation.
    :ivar step_size: Time step size for propagation.
    :ivar timesteps: How many discrete timesteps to propagate for.
    """

    def __init__(self, loggers: list[logging.Logger], step_size: float):
        """
        Pre-initialization, all these attributes (excluding step_size and loggers) are not filled in till setup() is
        called.
        """

        self.step_size = step_size
        self.loggers = loggers

        self.orbit = None
        self.perturbations = None
        self.final_time = None
        self.timesteps = None

    @abstractmethod
    def propagate(self):
        """
        Actual propagation is implemented here for child classes.

        NOTE: setup() MUST be called before this function in all circumstances.
        """

        pass

    def setup(
            self,
            orbit: orbit.Orbit,
            perturbations: list[perturbations.Perturbation],
            final_time: float,
    ):
        """
        Perform all the behind-the-scenes bookkeeping necessary to set up this object before propagating.
        """

        self.orbit = orbit
        self.perturbations = perturbations

        # Compute number of timesteps to propagate for and use this information to set up the Loggers.
        self.final_time = final_time
        if self.step_size is None:  # Default to 10000 steps.
            self.step_size = (final_time - self.orbit.time) / 10000

        self.timesteps = int(np.floor((self.final_time - orbit.time) / self.step_size))


    def log(self, timestep):
        """
        Store current information regarding the orbit.

        :param timestep: Current discrete timestep in propagation.
        """
        for logger in self.loggers:
            logger.log(propagator=self, timestep=timestep)
