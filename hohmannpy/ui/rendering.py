import pygfx as gfx
from rendercanvas.auto import RenderCanvas, loop
import numpy as np
from numpy.typing import NDArray
import importlib.resources
import imageio.v3 as iio
import pylinalg as la
from . import camera


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
            self.camera.elevation_dynamics_flag = 0
            self.camera.azimuth_dynamics_flag = 0
            self.camera.radial_dynamics_flag = 0

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
