import pygfx as gfx
from rendercanvas.auto import RenderCanvas, loop
import numpy as np
from numpy.typing import NDArray
import importlib.resources
import imageio.v3 as iio
import pylinalg as la


# TODO: This function is very WIP. It currently needs the following:
#   - Keyboard camera acceleration.
#   - Need to clean up all of the variables, such as having camera radius be based on central body radius and have it
#     be clamped between bounds.
class OrbitalCamera(gfx.PerspectiveCamera):
    def __init__(
            self,
            fov,
            aspect,
            radius,
            radial_accel,
            azimuth_accel,
            elevation_accel,
    ):
        self._azimuth = 0
        self._elevation = 0
        self.radius = radius

        self._azimuth_vel = 0
        self._elevation_vel = 0
        self._radial_vel = 0

        self.radial_accel_rate = radial_accel
        self.azimuth_accel_rate = azimuth_accel
        self.elevation_accel_rate = elevation_accel
        self.radial_accel = 0
        self.azimuth_accel = 0
        self.elevation_accel = 0

        super().__init__(fov, aspect)

    @property
    def azimuth(self):
        return self._azimuth
    @azimuth.setter
    def azimuth(self, value):
        value = value
        self._azimuth = value % 2 * np.pi

    @property
    def elevation(self):
        return self._elevation
    @elevation.setter
    def elevation(self, value):
        value = np.clip(value, -np.pi /2 + 1e-3, np.pi / 2 - 1e-3)
        self._elevation = value

    @property
    def radial_vel(self):
        return np.clip(self._radial_vel, -self.radial_accel_rate * 5, self.radial_accel_rate * 5)
    @radial_vel.setter
    def radial_vel(self, value):
        self._radial_vel = value
    @property
    def azimuth_vel(self):
        return np.clip(self._azimuth_vel, -self.azimuth_accel_rate * 5, self.azimuth_accel_rate * 5)
    @azimuth_vel.setter
    def azimuth_vel(self, value):
        self._azimuth_vel = value
    @property
    def elevation_vel(self):
        return np.clip(self._elevation_vel, -self.elevation_accel_rate * 5, self.elevation_accel_rate * 5)
    @elevation_vel.setter
    def elevation_vel(self, value):
        self._elevation_vel = value

    def orient(self):
        x, y, z = self.local.position

        radius = np.sqrt(x**2 + y**2 + z**2)
        if radius != 0:  # Safeguard because camera is oriented before mouse position is set.
            self.radius = radius
            self.elevation = np.arcsin(z / self.radius)
            self.azimuth = np.arctan2(y, x)

        self.elevation_vel += self.elevation_accel
        self.azimuth_vel += self.azimuth_accel
        self.radial_vel += self.radial_accel

        self.elevation += self.elevation_vel
        self.azimuth += self.azimuth_vel
        self.radius += self.radial_vel

        x = self.radius * np.cos(self.elevation) * np.cos(self.azimuth)
        y = self.radius * np.cos(self.elevation) * np.sin(self.azimuth)
        z = self.radius * np.sin(self.elevation)

        self.local.position = (x, y, z)
        self.show_pos((0, 0, 0), up=(0, 0, 1))

class RenderEngine:
    def __init__(self):
        earth_mat = gfx.MeshPhongMaterial()
        with importlib.resources.files("hohmannpy.resources").joinpath("earth_texture_map.jpg").open("rb") as f:
            earth_img = iio.imread(f)
            earth_img = np.ascontiguousarray(np.flipud(earth_img))
        earth_mat.map = gfx.Texture(earth_img, dim=2)

        self.central_body = gfx.Mesh(
            gfx.sphere_geometry(radius=6371, width_segments=64, height_segments=32),
            earth_mat
        )
        self.central_body.local.rotation = la.quat_from_euler((np.pi / 2, 0, 0), order="XYZ")

        self.canvas = RenderCanvas(size=(200, 200), title="TBD")
        self.renderer = gfx.renderers.WgpuRenderer(self.canvas)
        self.scene = gfx.Scene()
        self.scene.add(gfx.AmbientLight())
        self.scene.add(gfx.DirectionalLight())
        self.scene.add(self.central_body)
        x_axis, y_axis, z_axis = self.draw_basis()
        self.scene.add(x_axis)
        self.scene.add(y_axis)
        self.scene.add(z_axis)

        self.camera = OrbitalCamera(
            fov=50,
            aspect=16/9,
            radius=20000,
            radial_accel=1000,
            azimuth_accel=np.pi/24,
            elevation_accel=np.pi/36,
        )
        gfx.OrbitController(self.camera, register_events=self.renderer)

        self.canvas.add_event_handler(self.event_handler, "key_down", "key_up")

    def animate(self):
        self.camera.orient()
        self.renderer.render(self.scene, self.camera)
        self.canvas.request_draw(self.animate)

    def render(self):
        self.canvas.request_draw(self.animate)
        loop.run()

    def event_handler(self, event):
        if event["event_type"] == "key_down":
            key = event["key"].lower()
            match key:
                case "w":  # Rotate up.
                    self.camera.elevation_accel = self.camera.elevation_rate
                case "a":  # Rotate left.
                    self.camera.azimuth_vel = -self.camera.azimuth_rate
                case "s":  # Rotate down.
                    self.camera.elevation_vel = -self.camera.elevation_rate
                case "d":  # Rotate right.
                    self.camera.azimuth_vel = self.camera.azimuth_rate
                case "q":  # Zoom out.
                    self.camera.zoom_vel = self.camera.zoom_rate
                case "e":  # Zoom in.
                    self.camera.zoom_vel = -self.camera.zoom_rate
        elif event["event_type"] == "key_up":
            self.camera.elevation_vel = 0
            self.camera.azimuth_vel = 0
            self.camera.zoom_vel = 0

    def draw_eath(self):
        pass

    def draw_basis(self):
        length = 8000
        x_axis = gfx.Geometry(positions=np.array([[0, 0, 0], [length, 0, 0]], dtype=np.float32))
        y_axis = gfx.Geometry(positions=np.array([[0, 0, 0], [0, length, 0]], dtype=np.float32))
        z_axis = gfx.Geometry(positions=np.array([[0, 0, 0], [0, 0, length]], dtype=np.float32))
        x_material = gfx.LineMaterial(thickness=3, color=gfx.Color("#FF0000"))
        y_material = gfx.LineMaterial(thickness=3, color=gfx.Color("#00FF00"))
        z_material = gfx.LineMaterial(thickness=3, color=gfx.Color("#0000FF"))

        return gfx.Line(x_axis, x_material), gfx.Line(y_axis, y_material), gfx.Line(z_axis, z_material)

    def draw_orbit(self, traj: NDArray[float]):
        traj = traj.T / 1000
        traj = traj.astype(np.float32)
        self.scene.add(
            gfx.Line(gfx.Geometry(positions=traj), gfx.LineMaterial(thickness=2, color=gfx.Color("#FF073A")))
        )
