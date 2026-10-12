"""Three-camera 3-D rendering of a SuperDex scene: wrist, third-person and top-down views.

Uses SuperDex's own viewer (Polyscope) offscreen through headless EGL on the GPU. The FR3 and the 2F-85 are drawn from
their GLB visual meshes (registered like superdex.lab's mochi_helpers.create_bot) and recoloured like the real robot
(the viewer otherwise assigns a random colour to every link). Only ONE renderer can exist per process.

Positions are in metres in the world frame; images are uint8 RGB arrays.

Role in the system: ``scripts/render_hd.py`` uses this module's colour constants and :func:`register_bot_visuals`;
:class:`SceneRenderer` is the earlier, simpler three-view renderer.
"""

import numpy as np
import superdex.physics as physics
from superdex.physics.utils import render_model_registry
from superdex.physics.viewer.renderers.mesh_renderer import MeshRenderer
from superdex.physics.viewer.viewer import Viewer
from superdex.physics.viewer.viewer_cfg import ViewerCfg

# Colours (RGB in [0, 1]).
FRANKA_WHITE = [0.93, 0.93, 0.92]
FRANKA_DARK = [0.16, 0.16, 0.17]
GRIPPER_DARK = [0.12, 0.12, 0.13]
GRIPPER_PAD = [0.30, 0.30, 0.32]
PEG_WOOD = [0.62, 0.43, 0.25]
ROPE_COLOR = [0.85, 0.32, 0.12]
DARK_LINKS = ("fr3_link0", "fr3_link8")  # base plate and flange, dark grey on the real robot

# Fixed camera views (world coordinates [m]).
THIRD_PERSON = dict(look_from=[1.15, -0.95, 0.75], look_at=[0.30, -0.22, 0.08])
TOP = dict(look_from=[0.33, -0.22, 1.45], look_at=[0.33, -0.22, 0.0], up_dir=[1.0, 0.0, 0.0])


def quat_to_matrix(quaternion) -> np.ndarray:
    """Rotation matrix of a unit quaternion stored in (x, y, z, w) order.

    Examples
    --------
    >>> quat_to_matrix([0.0, 0.0, np.sin(np.pi / 4), np.cos(np.pi / 4)]).round(6).tolist()
    [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    """
    qx, qy, qz, qw = [float(value) for value in np.asarray(quaternion)]
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ])


def register_bot_visuals(scene, bot):
    """Register the bot's GLB render models with the scene (call before ``viewer.set_scene``)."""
    articulated = bot.get_articulated_actor()
    for handle, link in zip(articulated.get_nested_link_actors(), bot.get_bot_prefab().links):
        if link.render_model_file:
            transform = physics.TransformRT(link.render_model_rotation, link.render_model_translation)
            render_model_registry.register(scene, handle, link.render_model_file, transform, link.render_model_scale)


class SceneRenderer:
    """Renders wrist, third-person and top views of a scene that has ``.scene``, ``.bot`` and ``.rope_nodes()``.

    Parameters
    ----------
    sim : GraspScene
        The scene.
    size : tuple of int
        Image size (width, height) [px].
    wrist_link : str
        Link carrying the wrist camera (falls back to "fr3_link8").
    wrist_offset, wrist_look : sequence of float
        Wrist camera position and look-at point in the link frame [m].
    route : np.ndarray, shape (n, 2), optional
        Target route, drawn as a thin transparent line [m].
    """

    def __init__(self, sim, size=(480, 360), wrist_link="2f_85_base_link", wrist_offset=(0.0, -0.075, 0.035),
                 wrist_look=(0.0, 0.0, 0.16), route=None):
        self.sim = sim
        register_bot_visuals(sim.scene, sim.bot)
        self.viewer = Viewer(ViewerCfg(offscreen=True, size=tuple(size), coordinate_system="ros"))
        self.viewer.set_scene(sim.scene)
        self.viewer.render()  # create the actor renderers
        self._style()
        nodes = sim.rope_nodes()
        edges = np.stack([np.arange(len(nodes) - 1), np.arange(1, len(nodes))], 1)
        self.rope = self.viewer.add_curve_network("rope", nodes, edges, radius=sim.rope_radius, color=ROPE_COLOR)
        if route is not None:
            route_pts = np.c_[route, np.full(len(route), 0.0008)]
            self.viewer.add_curve_network("target_route", route_pts,
                                          np.stack([np.arange(len(route_pts) - 1), np.arange(1, len(route_pts))], 1),
                                          radius=0.0012, color=[0.25, 0.55, 0.25], transparency=0.5)
        info = sim.bot.get_articulated_actor().get_articulated_shape_info()
        self.link_names = list(info.link_names)
        self._buf = physics.DynamicArrayTransformRT(len(self.link_names))
        self.wrist_link = wrist_link if wrist_link in self.link_names else "fr3_link8"
        self.wrist_offset, self.wrist_look = np.array(wrist_offset), np.array(wrist_look)

    def _style(self):
        """Colour the robot links and the pegs like the real hardware, with a clay material."""
        for state in self.viewer._state.scene.actors.values():
            renderer = state.renderer
            if renderer is None or not hasattr(renderer, "set_front_face_color"):
                continue
            name = state.instance.get_name().split("/")[-1]  # link part (the bot name itself may contain "2f_85")
            if name.startswith("2f_85"):
                color = GRIPPER_PAD if "finger_tip" in name else GRIPPER_DARK
            elif name in DARK_LINKS:
                color = FRANKA_DARK
            elif name.startswith("fr3_"):
                color = FRANKA_WHITE
            elif name.startswith("peg"):
                color = PEG_WOOD
            else:
                continue
            try:
                renderer.set_front_face_color(color)
                renderer.set_material(MeshRenderer.Material.CLAY)
            except Exception:
                pass

    def _wrist_pose(self):
        """Camera view (look_from, look_at, up_dir) of the wrist camera at the current link pose."""
        self.sim.bot.get_articulated_actor().get_articulated_link_transforms(self._buf)
        transform = self._buf[self.link_names.index(self.wrist_link)]
        position = np.array(transform.translation, dtype=np.float64)
        rotation = quat_to_matrix(transform.rotation)
        eye = position + rotation @ self.wrist_offset
        target = position + rotation @ self.wrist_look
        up = rotation @ np.array([0.0, 1.0, 0.0])
        return dict(look_from=eye.tolist(), look_at=target.tolist(), up_dir=(-up).tolist())

    def render(self, views=("wrist", "third", "top")):
        """Render the requested views; returns a dict view name -> uint8 image (H, W, 3)."""
        self.rope.set_nodes(self.sim.rope_nodes())
        images = {}
        for view in views:
            camera = THIRD_PERSON if view == "third" else TOP if view == "top" else self._wrist_pose()
            self.viewer.set_camera_view(**camera)
            images[view] = np.asarray(self.viewer.render())[..., :3].copy()
        return images

    def panel(self, views=("wrist", "third", "top"), labels=True):
        """The requested views side by side in one image, optionally labelled (labels need OpenCV)."""
        images = self.render(views)
        row = np.concatenate([images[view] for view in views], axis=1)
        if labels:
            import cv2  # optional; skip labels if unavailable

            for index, view in enumerate(views):
                cv2.putText(row, {"wrist": "wrist camera", "third": "third-person", "top": "top"}[view],
                            (index * images[view].shape[1] + 10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 2)
        return row

    def close(self):
        """Close the viewer, ignoring errors."""
        try:
            self.viewer.close()
        except Exception:
            pass
