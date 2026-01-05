import pygfx as gfx
from rendercanvas.auto import RenderCanvas, loop
import numpy as np
from numpy.typing import NDArray


# TODO: This function is very WIP. It currently needs the following:
#   - Keyboard camera acceleration.
#   - Need to clean up all of the variables, such as having camera radius be based on central body radius and have it
#     be clamped between bounds.
#   - Potentially change azimuth clamping.
#   - Read up on pygfx documentation because some of this stuff is a black box at the moment.
class OrbitalCamera(gfx.PerspectiveCamera):
    def __init__(self, fov, aspect, radius, zoom_rate, azimuth_rate, elevation_rate):
        super().__init__(fov, aspect)

        self._azimuth = 0
        self._elevation = 0
        self.radius = radius

        self.azimuth_vel = 0
        self.elevation_vel = 0
        self.zoom_vel = 0

        self.zoom_rate = zoom_rate
        self.azimuth_rate = azimuth_rate
        self.elevation_rate = elevation_rate

    @property
    def azimuth(self):
        return self._azimuth
    @azimuth.setter
    def azimuth(self, value):
        value = value % (2 * np.pi)
        self._azimuth = value

    @property
    def elevation(self):
        return self._elevation
    @elevation.setter
    def elevation(self, value):
        value = np.clip(value, -np.pi/2 + 1e-3, np.pi/2 - 1e-3)
        self._elevation = value

    def orient(self):
        x, y, z = self.local.position

        radius = np.sqrt(x**2 + y**2 + z**2)
        if radius != 0:  # Safeguard because camera is oriented before mouse position is set.
            self.radius = radius
            self.elevation = np.arcsin(y / self.radius)
            self.azimuth = np.arctan2(x, z)

        self.elevation += self.elevation_vel
        self.azimuth += self.azimuth_vel
        self.radius += self.zoom_vel

        x = self.radius * np.cos(self.elevation) * np.sin(self.azimuth)
        y = self.radius * np.sin(self.elevation)
        z = self.radius * np.cos(self.elevation) * np.cos(self.azimuth)

        self.local.position = (x, y, z)
        self.show_pos((0, 0, 0))


class TempRenderEngine:
    def __init__(self):
        self.central_body = gfx.Mesh(
            gfx.sphere_geometry(radius=6371, width_segments=64, height_segments=32),
            gfx.MeshPhongMaterial(color="#0F52BA")
        )
        self.canvas = RenderCanvas(size=(200, 200), title="TBD")
        self.renderer = gfx.renderers.WgpuRenderer(self.canvas)
        self.scene = gfx.Scene()
        self.scene.add(gfx.AmbientLight())
        self.scene.add(gfx.DirectionalLight())
        self.scene.add(self.central_body)

        self.camera = OrbitalCamera(
            fov=50,
            aspect=16/9,
            radius=20000,
            zoom_rate=1000,
            azimuth_rate=np.pi/24,
            elevation_rate=np.pi/36
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
                    self.camera.elevation_vel = self.camera.elevation_rate
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

    def draw_orbit(self, traj: NDArray[float]):
        traj = traj.T / 1000
        traj = traj.astype(np.float32)
        self.scene.add(
            gfx.Line(gfx.Geometry(positions=traj), gfx.LineMaterial(thickness=2, color=gfx.Color("#FF073A")))
        )
