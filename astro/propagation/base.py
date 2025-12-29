from abc import ABC, abstractmethod
import numpy as np


class Propagator(ABC):
    """
    Base class for all propagators. All derivatives revolve around the method propagate() which takes in the initial
    orbital parameters (whatever those might be) and propagates them along the orbit up to the final time.

    :ivar orbit: Orbit to perform propagation on. Holds the initial conditions of the orbit including the starting time.
    :ivar final time: When to stop orbit propagation.
    :ivar step_size: Time step size for propagation.

    :ivar position_history: List of orbital positions (3, ) as a (3, length of propagation) array.
    :ivar velocity_history: List of orbital velocities (3, ) as a (3, length of propagation) array.
    :ivar time_history: List of time steps as a (1, length of propagation) array.
    """

    def __init__(self, orbit, final_time, step_size):
        self.orbit = orbit
        self.final_time = final_time
        self.step_size = step_size

        # Initialize history arrays. During propagation there will be N timesteps plus the initial timestep so the
        # arrays need to be of size N+1.
        timesteps = int(np.floor((self.final_time - orbit.time) / self.step_size))
        self.position_history = np.zeros([3, timesteps + 1])
        self.velocity_history = np.zeros([3, timesteps + 1])
        self.time_history = np.zeros([1, timesteps + 1])

        # Assign initial conditions to the history arrays.
        self.position_history[:, 0] = orbit.position
        self.velocity_history[:, 0] = orbit.velocity
        self.time_history[0, 0] = orbit.time

    @abstractmethod
    def propagate(self):
        pass
