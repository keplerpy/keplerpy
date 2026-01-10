from __future__ import annotations
import time

import numpy as np
import pygfx as gfx
import rendercanvas.auto
import importlib.resources
import imageio.v3 as iio
import pylinalg as la

from . import camera
from .. import astro


class RenderEngine:
    def __init__(
            self,
            traj: np.ndarray,
            initial_global_time: astro.Time,
            draw_basis: bool = True,
            draw_skybox: bool = True,
    ):
        self.initial_global_time = initial_global_time
        self.initial_local_time = None  # Set during initial animation.
        self.base_earth_rotation = None  # Set by draw_earth().

        # Create application window.
        self.canvas = rendercanvas.auto.RenderCanvas(size=(1280, 720), title="HohmannPy")
        self.renderer = gfx.renderers.WgpuRenderer(self.canvas)

        # Add lighting and objects to application.
        self.scene = gfx.Scene()
        self.scene.add(gfx.AmbientLight(intensity=0.1))
        sunlight = gfx.DirectionalLight(intensity=2)
        sunlight.local.position = (100000, 0, 0)
        self.scene.add(sunlight)

        self.earth = self.draw_earth()
        self.scene.add(self.earth)

        self.orbit = self.draw_orbit(traj)
        self.scene.add(self.orbit)

        if draw_basis:
            x_axis, y_axis, z_axis = self.draw_basis()
            self.scene.add(x_axis)
            self.scene.add(y_axis)
            self.scene.add(z_axis)

        if draw_skybox:
            skybox = self.draw_skybox()
            self.scene.add(skybox)

        # Create the camera.
        self.camera = camera.OrbitalCamera(
            fov=50,
            aspect=16/9,
            initial_radius=20000,
            min_radius=9000,
            radial_accel=50000,
            azimuth_accel=3 * np.pi / 2,
            elevation_accel=3 * np.pi / 2,
            radial_damping=100000,
            azimuth_damping=4 * np.pi,
            elevation_damping=4 * np.pi,
            max_radial_vel=50000,
            max_azimuth_vel=2 * np.pi,
            max_elevation_vel=2 * np.pi,
        )
        gfx.OrbitController(self.camera, register_events=self.renderer)  # Add mouse control.

        # Add event dispatch functionality.
        self.canvas.add_event_handler(self.event_handler, "key_down", "key_up")

    def animate(self):
        earth_rot = 7.292115e-5  # Mean rotation rate of the Earth in radians.
        self.camera.orient()
        self.earth.local.rotation = la.quat_mul(
            self.base_earth_rotation,
            la.quat_from_axis_angle((0, 1, 0), -(time.perf_counter() - self.initial_local_time) * earth_rot),
        )
        self.renderer.render(self.scene, self.camera)
        self.canvas.request_draw(self.animate)

    def render(self):
        self.initial_local_time = time.perf_counter()
        self.canvas.request_draw(self.animate)
        rendercanvas.auto.loop.run()

    def event_handler(self, event):
        if event["event_type"] == "key_down":
            key = event["key"].lower()
            match key:
                case "w":  # Rotate up.
                    self.camera.elevation_dynamics_flag = 1
                case "a":  # Rotate left.
                    self.camera.azimuth_dynamics_flag = -1
                case "s":  # Rotate down.
                    self.camera.elevation_dynamics_flag = -1
                case "d":  # Rotate right.
                    self.camera.azimuth_dynamics_flag = 1
                case "q":  # Zoom out.
                    self.camera.radial_dynamics_flag = 1
                case "e":  # Zoom in.
                    self.camera.radial_dynamics_flag = -1
        else:
            if event["event_type"] == "key_up":
                key = event["key"].lower()
                match key:
                    case "w":  # Rotate up.
                        self.camera.elevation_dynamics_flag = 0
                    case "a":  # Rotate left.
                        self.camera.azimuth_dynamics_flag = 0
                    case "s":  # Rotate down.
                        self.camera.elevation_dynamics_flag = 0
                    case "d":  # Rotate right.
                        self.camera.azimuth_dynamics_flag = 0
                    case "q":  # Zoom out.
                        self.camera.radial_dynamics_flag = 0
                    case "e":  # Zoom in.
                        self.camera.radial_dynamics_flag = 0

    def draw_earth(self):
        # Initialize the Earth texture.
        earth_mat = gfx.MeshPhongMaterial(shininess=5)
        with importlib.resources.files("hohmannpy.resources").joinpath("earth_texture_map.jpg").open("rb") as f:
            earth_img = iio.imread(f)
            earth_img = np.ascontiguousarray(np.flipud(earth_img))  # Need to flip array.
        earth_mat.map = gfx.Texture(earth_img, dim=2)

        # Create the Earth object using the texture.
        earth = gfx.Mesh(
            gfx.sphere_geometry(radius=6371, width_segments=64, height_segments=32),
            earth_mat
        )

        self.base_earth_rotation = la.quat_from_euler(
            (np.pi / 2, -self.initial_global_time.gmst, 0), order="XYZ"
        ) # Rotate Earth since texture is 90 deg offset about x-axis, then offset terminator in new body frame.
        earth.local.rotation = self.base_earth_rotation

        return earth

    def draw_skybox(self):
        # Import all six skybox faces.
        with importlib.resources.files("hohmannpy.resources").joinpath("skybox/skybox_right1.png").open("rb") as f:
            skybox_right1_img = iio.imread(f)
        with importlib.resources.files("hohmannpy.resources").joinpath("skybox/skybox_left2.png").open("rb") as f:
            skybox_left2_img = iio.imread(f)
        with importlib.resources.files("hohmannpy.resources").joinpath("skybox/skybox_top3.png").open("rb") as f:
            skybox_top3_img = iio.imread(f)
        with importlib.resources.files("hohmannpy.resources").joinpath("skybox/skybox_bottom4.png").open("rb") as f:
            skybox_bottom4_img = iio.imread(f)
        with importlib.resources.files("hohmannpy.resources").joinpath("skybox/skybox_front5.png").open("rb") as f:
            skybox_front5_img = iio.imread(f)
        with importlib.resources.files("hohmannpy.resources").joinpath("skybox/skybox_back6.png").open("rb") as f:
            skybox_back6_img = iio.imread(f)

        # Stack the faces.
        skybox_img = np.stack(
            [skybox_right1_img, skybox_left2_img, skybox_top3_img,
             skybox_bottom4_img, skybox_front5_img, skybox_back6_img],
            axis=0
        )

        # Create the skybox object.
        width = skybox_img.shape[1]
        height = skybox_img.shape[2]
        skybox = gfx.Background(
            None,
            gfx.BackgroundSkyboxMaterial(map=gfx.Texture(skybox_img, dim=2, size=(width, height, 6))),
        )

        return skybox

    def draw_basis(self):
        length = 8000

        x_axis = gfx.Geometry(positions=np.array([[0, 0, 0], [length, 0, 0]], dtype=np.float32))
        y_axis = gfx.Geometry(positions=np.array([[0, 0, 0], [0, length, 0]], dtype=np.float32))
        z_axis = gfx.Geometry(positions=np.array([[0, 0, 0], [0, 0, length]], dtype=np.float32))

        x_material = gfx.LineMaterial(thickness=3, color=gfx.Color("#FF0000"))
        y_material = gfx.LineMaterial(thickness=3, color=gfx.Color("#00FF00"))
        z_material = gfx.LineMaterial(thickness=3, color=gfx.Color("#0000FF"))

        return gfx.Line(x_axis, x_material), gfx.Line(y_axis, y_material), gfx.Line(z_axis, z_material)

    def draw_orbit(self, traj: np.ndarray):
        orbit = traj.T / 1000  # Scale to engine units (km).
        orbit = orbit.astype(np.float32)  # Data type needed by gfx.Geometry.

        return gfx.Line(gfx.Geometry(positions=orbit), gfx.LineMaterial(thickness=2, color=gfx.Color("#FF073A")))
