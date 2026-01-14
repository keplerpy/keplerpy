from __future__ import annotations
import time

import numpy as np
import pygfx as gfx


class OrbitalCamera(gfx.PerspectiveCamera):
    r"""
    A custom orbital camera built off the pygfx library with keyboard control.

    Cameras in pygfx are seperated into two submodules: :mod:`pygfx.cameras` and :mod:`pygfx.controllers`. The former
    determines the view style (such as 2D vs 3D) and the latter how the camera moves in response to user input. To
    create an orbit camera (a camera which rotates around an object like it is orbiting it)
    :class:`pygfx.PerspectiveCamera` may be passed into :class:`pygfx.OrbitController`. However, this only implements
    mouse-based control.

    To remedy this, this class may be used. It is a child class of :class:`pygfx.PerspectiveCamera` and unlike a typical
    class from :mod:`pygfx.cameras` it implements its own movement dynamics via :meth:`orient()`
    without the need for a :mod:`pygfx.controller`. However, in-order for user movement commands to be processed this
    class must be used in conjunction with :class:`ui.RenderEngine` which is responsible for converting user input into
    commands readable by :meth:`orient()`. The camera's equations of motion are displayed in the Notes
    section of :meth:`orient()`.

    An orbit camera uses spherical coordinates (azimuth, elevation, and radius) to position itself with respect to
    a fixed central point, in this case :math`[x, y, z] = [0, 0, 0]` in the :class:`ui.RenderEngine`'s Cartesian
    coordinates, which the camera always points directly towards. Changing each of these has the following effects:

    - **radius**: moves the camera towards or away from the sphere. Represents the magnitude of the position vector between the camera and :math:`[0, 0, 0]`.

    - **azimuth**: rotates the camera left and right about the sphere. Represents the angle between the x-axis and a projection of the camera's radius into the xy-plane.

    - **elevation**: rotates the camera up and down about the sphere. Represents the angle between the projection of the camera's radius into the xy-plane and the position vector itself.

    Parameters
    ----------
    fov: float
        The field of view of the camera.
    aspect: float
        The aspect ratio of the camera.
    initial_radius: float
        The distance of the camera from :math:`[0, 0, 0]` upon instantiation.
    min_radius: float
        The minimum distance between the camera and :math:`[0, 0, 0]`.
    radial_accel: float
        Radial acceleration of the camera (zoom rate) when the user moves the camera in or out.
    azimuth_accel: float
        Left-right rotational acceleration of the camera when the user rotates the camera.
    elevation_accel: float
        Up-down rotational acceleration of the camera when the user rotates the camera.
    radial_damping: float
        How fast radial velocity is killed when user input stops.
    azimuth_damping: float
        How fast azimuthal velocity is killed when user input stops.
    elevation_damping: float
        How fast elevational velocity is killed when user input stops.
    max_radial_vel: float
        Maximum radial velocity of the camera.
    max_azimuth_vel: float
        Maximum azimuthal velocity of the camera.
    max_elevation_vel: float
        Maximum elevation velocity of the camera.

    Attributes
    ----------
    min_radius: float
        The minimum distance between the camera and :math:`[0, 0, 0]`.
    radial_accel: float
        Radial acceleration of the camera (zoom rate) when the user moves the camera in or out.
    azimuth_accel: float
        Left-right rotational acceleration of the camera when the user rotates the camera.
    elevation_accel: float
        Up-down rotational acceleration of the camera when the user rotates the camera.
    radial_damping: float
        How fast radial velocity is killed when user input stops.
    azimuth_damping: float
        How fast azimuthal velocity is killed when user input stops.
    elevation_damping: float
        How fast elevational velocity is killed when user input stops.
    max_radial_vel: float
        Maximum radial velocity of the camera.
    max_azimuth_vel: float
        Maximum azimuthal velocity of the camera.
    max_elevation_vel: float
        Maximum elevation velocity of the camera.
    radial_dynamics_flag: int
        Indicates current user zoom command. 0 = no zoom, -1 = currently zooming in, 1 = currently zooming out.
    azimuth_dynamics_flag: int
        Indicates current user azimuthal rotation command. 0 = no rotation, -1 = currently rotating left, 1 = currently
        rotating right.
    elevation_dynamics_flag: int
        Indicates current user elevation rotation command. 0 = no rotation, -1 = currently rotating up, 1 = currently
        rotating down.
    stored_time: float
        Global timestamp at the last time :meth:`orient()` was called.
    """

    def __init__(
            self,
            fov: float,
            aspect: float,
            initial_radius: float,
            min_radius: float,
            radial_accel: float,
            azimuth_accel: float,
            elevation_accel: float,
            radial_damping: float,
            azimuth_damping: float,
            elevation_damping: float,
            max_radial_vel: float,
            max_azimuth_vel: float,
            max_elevation_vel: float,
    ):
        # Initial dynamics. Camera status at rest at [radius, azimuth, elevation] = [initial_radius, 0, 0].
        self.radius = initial_radius
        self.min_radius = min_radius
        self.radial_vel = 0
        self.radial_accel = radial_accel
        self.radial_damping = radial_damping
        self.max_radial_vel = max_radial_vel

        self.azimuth = 0
        self.azimuth_vel = 0
        self.azimuth_accel = azimuth_accel
        self.azimuth_damping = azimuth_damping
        self.max_azimuth_vel = max_azimuth_vel

        self.elevation = 0
        self.elevation_vel = 0
        self.elevation_accel = elevation_accel
        self.elevation_damping = elevation_damping
        self.max_elevation_vel = max_elevation_vel

        self.radial_dynamics_flag: int = 0
        self.azimuth_dynamics_flag: int = 0
        self.elevation_dynamics_flag: int = 0

        self.stored_time: float = time.perf_counter()

        super().__init__(fov, aspect)

    def orient(self):
        r"""
        Implements the equations of motion for the camera.

        Takes in the current camera position in the frame of :class:`ui.RenderEngine` as Cartesian coordinates and
        converts these to spherical coordinates. These coordinates are then adjusted based on user input before being
        clamped and converted back to a new set of Carterian which is used to update the camera's position.

        Notes
        -----
        The equations of motion for the camera are computed in spherical coordinates [1]_ in which co-elevation is
        replaced with elevation. The structure of the equations is the same for azimuth :math:`\theta`, elevation
        :math:`\phi`, and radius :math:`\rho` so just azimuth is shown below as an example. The exact equation used
        depends on the current value of :attr:`azimuth_dynamics_flag` :math:`\text{FLAG}`.

        .. math::

            \text{FLAG} =
            \begin{cases}
                1,  & \dot{\theta}_{t+\Delta t} = \dot{\theta}_{t} + \ddot{\theta} \Delta t \\
                0,  & \dot{\theta}_{t+\Delta t} = \dot{\theta}_{t} e^{-\zeta_\theta \Delta t} \\
                -1, & \dot{\theta}_{t+\Delta t} = \dot{\theta}_{t} - \ddot{\theta} \Delta t
            \end{cases}

        .. math::

            \theta_{t+\Delta t} = \theta_{t}
                \text{clip}(\dot{\theta}_{t+\Delta t}, -\dot{\theta}_\max, \dot{\theta}_\max) \Delta t

        where :math:`\zeta_\theta`, :math:`\ddot{\theta}`, :math:`\dot{\theta}_\max` are :attr:`azimuth_damping`,
        :attr:`azimuth_accel`, and :attr:`max_azimuth_vel` respectively. These equations are discrete because their
        implementation in :meth:`orient()` is not called continuously but rather every :math:`\Delta t` seconds.

        In addition, the following bounds are placed on the state variables:

        .. math::
            \begin{aligned}
                0 &\le \theta < 2\pi \\
                -\tfrac{\pi}{2} + \varepsilon
                  &\le \phi
                  \le \tfrac{\pi}{2} - \varepsilon \\
                \rho_{\min}
                  &\le \rho < \infty
            \end{aligned}

        where :math:`\dot{\theta}_\min` is :attr:`min_radius`. :math:`\varepsilon = 1e^{-3}` is used to avoid numerical
        singularities that would occur when :math:`\phi = -\pi/2, \, \pi / 2` exactly.

        References
        ----------
        .. [1] LibreTexts, “12.7: Cylindrical and Spherical Coordinates,” in Calculus (OpenStax), Math LibreTexts.
        """

        # Retrieve the position from the camera. These can't be stored locally be mouse movement via
        # pygfx.OrbitController may have been used to move the camera between calls to this function.
        x, y, z = self.local.position

        # Compute radius, elevation, and azimuth.
        radius = np.sqrt(x**2 + y**2 + z**2)
        if radius != 0:  # Safeguard because camera is oriented before mouse position is set.
            self.radius = radius
            self.elevation = np.arcsin(z / self.radius)
            self.azimuth = np.arctan2(y, x)

        # Evaluate dynamics.
        time_change = time.perf_counter() - self.stored_time
        self.stored_time = time.perf_counter()
        match self.radial_dynamics_flag:
            case 0:
                self.radial_vel *= np.exp(-self.radial_damping * time_change)
            case 1:
                self.radial_vel += self.radial_accel * time_change
            case -1:
                self.radial_vel -= self.radial_accel * time_change
        self.radial_vel = np.clip(self.radial_vel, -self.max_radial_vel, self.max_radial_vel)
        match self.azimuth_dynamics_flag:
            case 0:
                self.azimuth_vel *= np.exp(-self.azimuth_damping * time_change)
            case 1:
                self.azimuth_vel += self.azimuth_accel * time_change
            case -1:
                self.azimuth_vel -= self.azimuth_accel * time_change
        self.azimuth_vel = np.clip(self.azimuth_vel, -self.max_azimuth_vel, self.max_azimuth_vel)
        match self.elevation_dynamics_flag:
            case 0:
                self.elevation_vel *= np.exp(-self.elevation_damping * time_change)
            case 1:
                self.elevation_vel += self.elevation_accel * time_change
            case -1:
                self.elevation_vel -= self.elevation_accel * time_change
        self.elevation_vel = np.clip(self.elevation_vel, -self.max_elevation_vel, self.max_elevation_vel)

        self.elevation += self.elevation_vel * time_change
        self.azimuth += self.azimuth_vel * time_change
        self.radius += self.radial_vel * time_change

        # Clamp radius, azimuth, and elevation.
        self.radius = np.max([self.radius, self.min_radius])
        self.azimuth %= 2 * np.pi

        if self.elevation <= -np.pi / 2 + 1e-3:
            self.elevation = -np.pi / 2 + 1e-3
            self.elevation_vel = 0
        if self.elevation >= np.pi / 2 - 1e-3:
            self.elevation = np.pi / 2 - 1e-3
            self.elevation_vel = 0

        # Update Cartesian position and of the camera.
        x = self.radius * np.cos(self.elevation) * np.cos(self.azimuth)
        y = self.radius * np.cos(self.elevation) * np.sin(self.azimuth)
        z = self.radius * np.sin(self.elevation)
        self.local.position = (x, y, z)

        # Point towards [0, 0, 0] with the z-axis as up. This last setting controls the camera's roll-orientation.
        self.show_pos((0, 0, 0), up=(0, 0, 1))
