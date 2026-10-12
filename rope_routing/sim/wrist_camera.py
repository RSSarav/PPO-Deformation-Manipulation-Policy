"""Wrist camera for SuperDex scenes (FR3 rope routing).

What SuperDex provides (checked in project_superdex, 2026-10-11):

* superdex.robotics has a native camera *sensor*, type "SENSOR_CAMERA" (bots.CameraSensor /
  bots.CameraSensorParams; superdex_robotics/include/superdex_robotics/sensors/camera_sensor.h). It is attached to a
  bot link and stores the mount and the intrinsics, but it produces NO pixels: "renderers query GetParams() to
  configure their camera actors" (camera_sensor.h:81-84).
* The renderers meant to consume it need external servers that do not exist on this machine: UnrealCVViewer (Unreal
  Engine with the UnrealCV plugin) and MochiRendererViewer (a standalone ``mochi_viewer`` TCP server, not in the
  repository and not built).
* SuperDex's own Polyscope ``Viewer`` works headless (EGL on the GPU) and draws the real FR3 GLB link meshes from the
  live scene. It has one global camera (set_camera_view and the Polyscope field of view) and returns RGBA only.

This module therefore creates a real SENSOR_CAMERA on the wrist link (its pose and intrinsics live in SuperDex and
follow the link), renders its view with SuperDex's Viewer, and reads metric depth from the Viewer's OpenGL depth
buffer.

Conventions:

* The mount is given in the link frame (CameraSensorParams semantics): eye = link * offset_local, look direction =
  R_link * forward_axis, image up = R_link * up_axis_local. For fr3_link8, +z is the tool axis. The environment
  commands the tool down (Rx(pi)), so link8 +x = world +x (away from the base, toward the pegs) and link8 +z =
  world -z.
* ``world_from_cam()`` returns the OpenCV optical frame (x right, y down, z forward). ``intrinsics`` uses the OpenCV
  pixel-index convention (pixel centres at integers): fx = fy = (H/2) / tan(fov_v / 2), cx = (W - 1) / 2,
  cy = (H - 1) / 2 for an image of W x H pixels.
* Depth is the z-depth along the optical axis [m], float32, inf where nothing was hit or beyond far_clip.
* Polyscope's near and far planes are near = 0.005 L and far = 20 L, where L = ps.get_length_scale() (the Viewer sets
  L = 8, so near = 4 cm). ``render()`` temporarily sets L = near_clip / 0.005. Do NOT use ps.set_view_from_json to
  change the clip ratios: it also moves the camera (checked).

Positions are in metres, angles in degrees where named ``*_deg``, images in pixels.

Role in the system: ``scripts/render_hd.py`` uses :class:`WristCamera` with ``MOUNTS["2f85_side"]`` for the wrist view
of the videos. :func:`make_viewer` and :class:`WristCameraObs` belong to an earlier rope scene (``sim.rp``) and are
kept for reference; the RL environments do not use images.
"""

from __future__ import annotations

import ctypes
import fnmatch
import json

import numpy as np
import polyscope as ps
import superdex.physics as physics
import superdex.robotics as robotics
from scipy.spatial.transform import Rotation
from superdex.physics.utils import render_model_registry

__all__ = ["WristCamera", "WristCameraObs", "MOUNTS", "register_visuals", "tf_to_mat", "make_viewer"]

# Mounts on fr3_link8 (+z = tool axis). tilt_deg tilts the optical axis from +z toward the tool axis.
MOUNTS = {
    # Arm without gripper (tool down = Rx(pi)): 7 cm behind the flange axis on the robot-base side, looking forward and
    # down; the tool point (rope end, 10 cm below the flange) sits just below the image centre and the area ahead
    # (+x, toward the pegs) fills the upper half.
    "flange_back": dict(link_name="fr3_link8", offset_local=(-0.07, 0.0, 0.02), tilt_deg=45.0, fov_vertical_deg=65.0),
    # fr3_v2_2f_85: the 2F-85 fingers open along link8 x, so the camera sits on the +y side (perpendicular to the
    # finger motion, both fingers visible), 10 cm off-axis and 5 cm below the flange, tilted 40 degrees inward (a
    # closed grasp 16 cm below the flange lands at the image centre).
    "2f85_side": dict(link_name="fr3_link8", offset_local=(0.0, 0.10, 0.05), tilt_deg=40.0, fov_vertical_deg=65.0),
}


def tf_to_mat(transform) -> np.ndarray:
    """4 x 4 homogeneous matrix of a SuperDex TransformRT (its quaternion's tolist() is x, y, z, w)."""
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_quat(transform.rotation.tolist()).as_matrix()
    matrix[:3, 3] = np.asarray(transform.translation, dtype=np.float64)
    return matrix


def register_visuals(sim) -> None:
    """Register the bot's GLB render models (call before viewer.set_scene); ``render.register_bot_visuals`` does the
    same."""
    articulated = sim.bot.get_articulated_actor()
    for handle, link in zip(articulated.get_nested_link_actors(), sim.bot.get_bot_prefab().links):
        if link.render_model_file:
            transform = physics.TransformRT(link.render_model_rotation, link.render_model_translation)
            render_model_registry.register(sim.scene, handle, link.render_model_file, transform,
                                           link.render_model_scale)


# ------------------------------------------------------------------------------------------------- OpenGL depth
class _GLDepthReader:
    """Reads Polyscope's scene depth buffer through ctypes, in the EGL context that Polyscope made current.

    The scene framebuffer is the one with an RGBA16F colour texture and a depth texture (Polyscope's sceneBuffer).
    Only the READ framebuffer binding, the texture binding and GL_PACK_ALIGNMENT are touched, and all are restored.
    """

    _GL_READ_FRAMEBUFFER, _GL_READ_FRAMEBUFFER_BINDING = 0x8CA8, 0x8CAA
    _GL_COLOR_ATTACHMENT0, _GL_DEPTH_ATTACHMENT = 0x8CE0, 0x8D00
    _OBJ_TYPE, _OBJ_NAME, _GL_TEXTURE = 0x8CD0, 0x8CD1, 0x1702
    _GL_TEXTURE_2D, _GL_TEXTURE_BINDING_2D = 0x0DE1, 0x8069
    _GL_TEXTURE_WIDTH, _GL_TEXTURE_HEIGHT, _GL_TEXTURE_INTERNAL_FORMAT = 0x1000, 0x1001, 0x1003
    _GL_RGBA16F, _GL_PACK_ALIGNMENT, _GL_DEPTH_COMPONENT, _GL_FLOAT = 0x881A, 0x0D05, 0x1902, 0x1406

    def __init__(self):
        egl = ctypes.CDLL("libEGL.so.1")
        egl.eglGetProcAddress.restype = ctypes.c_void_p
        egl.eglGetProcAddress.argtypes = [ctypes.c_char_p]
        egl.eglGetCurrentContext.restype = ctypes.c_void_p
        if not egl.eglGetCurrentContext():
            raise RuntimeError("no current EGL context: create the superdex Viewer (offscreen, headless EGL) first")

        def fn(name, restype, *argtypes):
            addr = egl.eglGetProcAddress(name.encode())
            if not addr:
                raise RuntimeError(f"eglGetProcAddress({name}) failed")
            return ctypes.CFUNCTYPE(restype, *argtypes)(addr)

        c_uint, c_int, c_int_ptr = ctypes.c_uint, ctypes.c_int, ctypes.POINTER(ctypes.c_int)
        self.IsFramebuffer = fn("glIsFramebuffer", ctypes.c_ubyte, c_uint)
        self.BindFramebuffer = fn("glBindFramebuffer", None, c_uint, c_uint)
        self.GetIntegerv = fn("glGetIntegerv", None, c_uint, c_int_ptr)
        self.GetFBAttach = fn("glGetFramebufferAttachmentParameteriv", None, c_uint, c_uint, c_uint, c_int_ptr)
        self.BindTexture = fn("glBindTexture", None, c_uint, c_uint)
        self.GetTexLevelParameteriv = fn("glGetTexLevelParameteriv", None, c_uint, c_int, c_uint, c_int_ptr)
        self.ReadPixels = fn("glReadPixels", None, c_int, c_int, c_int, c_int, c_uint, c_uint, ctypes.c_void_p)
        self.PixelStorei = fn("glPixelStorei", None, c_uint, c_int)
        self.fbo, self.depth_tex = None, None

    def _geti(self, pname) -> int:
        """glGetIntegerv for a single integer."""
        value = ctypes.c_int(0)
        self.GetIntegerv(pname, ctypes.byref(value))
        return value.value

    def _attachment(self, att):
        """Texture name attached to ``att`` of the bound read framebuffer, or None if it is not a texture."""
        obj_type, obj_name = ctypes.c_int(0), ctypes.c_int(0)
        self.GetFBAttach(self._GL_READ_FRAMEBUFFER, att, self._OBJ_TYPE, ctypes.byref(obj_type))
        if obj_type.value != self._GL_TEXTURE:
            return None
        self.GetFBAttach(self._GL_READ_FRAMEBUFFER, att, self._OBJ_NAME, ctypes.byref(obj_name))
        return obj_name.value

    def _tex_info(self, tex):
        """(width, height, internal format) of a 2-D texture."""
        old = self._geti(self._GL_TEXTURE_BINDING_2D)
        self.BindTexture(self._GL_TEXTURE_2D, tex)
        width, height, fmt = ctypes.c_int(0), ctypes.c_int(0), ctypes.c_int(0)
        self.GetTexLevelParameteriv(self._GL_TEXTURE_2D, 0, self._GL_TEXTURE_WIDTH, ctypes.byref(width))
        self.GetTexLevelParameteriv(self._GL_TEXTURE_2D, 0, self._GL_TEXTURE_HEIGHT, ctypes.byref(height))
        self.GetTexLevelParameteriv(self._GL_TEXTURE_2D, 0, self._GL_TEXTURE_INTERNAL_FORMAT, ctypes.byref(fmt))
        self.BindTexture(self._GL_TEXTURE_2D, old)
        return width.value, height.value, fmt.value

    def _find_scene_fbo(self):
        """Find Polyscope's scene framebuffer (RGBA16F colour and depth textures) and remember it."""
        old = self._geti(self._GL_READ_FRAMEBUFFER_BINDING)
        found = None
        try:
            for fb in range(1, 512):
                if not self.IsFramebuffer(fb):
                    continue
                self.BindFramebuffer(self._GL_READ_FRAMEBUFFER, fb)
                color, depth = self._attachment(self._GL_COLOR_ATTACHMENT0), self._attachment(self._GL_DEPTH_ATTACHMENT)
                if color is not None and depth is not None and self._tex_info(color)[2] == self._GL_RGBA16F:
                    found = (fb, depth)
                    break
        finally:
            self.BindFramebuffer(self._GL_READ_FRAMEBUFFER, old)
        if found is None:
            raise RuntimeError("could not find Polyscope's scene framebuffer (RGBA16F color + depth texture)")
        self.fbo, self.depth_tex = found

    def read_raw(self) -> np.ndarray:
        """Window-space depth in [0, 1], rows top-down, at the scene-buffer resolution (SSAA x window)."""
        if self.fbo is None:
            self._find_scene_fbo()
        width, height, _ = self._tex_info(self.depth_tex)
        old_fb, old_align = self._geti(self._GL_READ_FRAMEBUFFER_BINDING), self._geti(self._GL_PACK_ALIGNMENT)
        out = np.empty((height, width), np.float32)
        self.BindFramebuffer(self._GL_READ_FRAMEBUFFER, self.fbo)
        self.PixelStorei(self._GL_PACK_ALIGNMENT, 1)
        self.ReadPixels(0, 0, width, height, self._GL_DEPTH_COMPONENT, self._GL_FLOAT, out.ctypes.data)
        self.PixelStorei(self._GL_PACK_ALIGNMENT, old_align)
        self.BindFramebuffer(self._GL_READ_FRAMEBUFFER, old_fb)
        return out[::-1]

    @staticmethod
    def clip_planes():
        """(near, far) of Polyscope's perspective projection [m]: clip ratio times length scale (verified by
        regression)."""
        view = json.loads(ps.get_view_as_json())
        scale = ps.get_length_scale()
        return view["nearClipRatio"] * scale, view["farClipRatio"] * scale

    def read_metric(self, width: int, height: int) -> np.ndarray:
        """Metric z-depth [m] at the requested image size (inf where nothing was hit), float32 (height, width).

        The depth buffer is linearised with the glm::perspective mapping. With supersampling (SSAA), k x k samples
        are pooled per pixel by averaging 1/z, which is exact for planes; an unexpected buffer size falls back to the
        nearest sample at each pixel centre.
        """
        depth = self.read_raw().astype(np.float64)
        near, far = self.clip_planes()
        # Inverse of glm::perspective's depth mapping (NDC z in [-1, 1]).
        z_depth = 2.0 * near * far / ((far + near) - (far - near) * (2.0 * depth - 1.0))
        z_depth[depth >= 1.0] = np.inf
        buf_height, buf_width = z_depth.shape
        if (buf_width, buf_height) != (width, height):
            factor = buf_height // height
            if factor * height == buf_height and factor * width == buf_width:  # SSAA: pool factor x factor samples
                inv = (1.0 / z_depth).reshape(height, factor, width, factor).mean(axis=(1, 3))
                z_depth = np.where(inv > 0, 1.0 / np.maximum(inv, 1e-12), np.inf)
            else:  # unexpected buffer size: nearest sample at each pixel centre
                rows = ((np.arange(height) + 0.5) * buf_height / height).astype(int)
                cols = ((np.arange(width) + 0.5) * buf_width / width).astype(int)
                z_depth = z_depth[np.ix_(rows, cols)]
        return z_depth.astype(np.float32)


# ------------------------------------------------------------------------------------------------- camera
class WristCamera:
    """A SENSOR_CAMERA on a bot link, rendered with SuperDex's Polyscope Viewer.

    Parameters
    ----------
    bot
        superdex.robotics Bot (e.g. ``env.sim.bot``).
    viewer
        superdex.physics.viewer.viewer.Viewer, offscreen, already ``set_scene(...)`` on the bot's scene.
    link_name : str
        Link the camera is mounted on ("fr3_link8"; for fr3_v2_2f_85 also fr3_link8 or "2f_85_base_link").
    width, height : int
        Image size [px] (square pixels, principal point at the image centre).
    fov_vertical_deg : float
        Vertical field of view [deg].
    offset_local : sequence of float
        Camera position in the link frame [m].
    tilt_deg : float
        Optical-axis tilt from the link +z axis toward the tool axis, i.e. toward -offset_local[:2] (toward +x if the
        offset is on the axis) [deg]. Image up is the in-plane direction toward the flange side. Ignored if
        ``forward_axis`` is given (``up_axis_local`` is then required).
    forward_axis, up_axis_local : sequence of float, optional
        Explicit look direction and image-up direction in the link frame.
    near_clip, far_clip : float
        Near clip plane; depth beyond far_clip is reported as inf [m].
    name : str
        Sensor name.
    hide : sequence of str
        Viewer curve networks hidden in the wrist view (fnmatch patterns): by default the "target" route overlay and
        all camera frustum overlays.
    restore_transparency_mode : str or None
        The wrist view is rendered with ps.set_transparency_mode("none") (a single pass, so the depth buffer holds the
        nearest surface; in "pretty" mode Polyscope depth-peels and the buffer ends up holding the last peel layer).
        Polyscope has no getter for the mode, so this is the mode set back afterwards: "auto" means "pretty" if any
        viewer curve network or point cloud is semi-transparent (Polyscope switches to "pretty" by itself when a
        structure gets transparency < 1), else "none"; None leaves "none".

    Examples
    --------
    >>> camera = WristCamera(env.sim.bot, viewer, width=640, height=480, **MOUNTS["2f85_side"])  # doctest: +SKIP
    >>> rgb, depth = camera.render(depth=True)  # doctest: +SKIP
    >>> rgb.shape, depth.shape, camera.intrinsics.shape  # doctest: +SKIP
    ((480, 640, 3), (480, 640), (3, 3))
    """

    def __init__(self, bot, viewer, link_name: str = "fr3_link8", width: int = 128, height: int = 128,
                 fov_vertical_deg: float = 70.0, offset_local=(-0.07, 0.0, 0.02), tilt_deg: float = 40.0,
                 forward_axis=None, up_axis_local=None, near_clip: float = 0.01, far_clip: float = 3.0,
                 name: str = "wrist_cam", hide=("target", "*_frustum"), restore_transparency_mode: str | None = "auto"):
        if forward_axis is None:
            tilt = np.deg2rad(tilt_deg)
            inward = -np.array([offset_local[0], offset_local[1], 0.0], dtype=float)
            inward = inward / np.linalg.norm(inward) if np.linalg.norm(inward) > 1e-9 else np.array([1.0, 0.0, 0.0])
            z_axis = np.array([0.0, 0.0, 1.0])
            forward_axis = np.sin(tilt) * inward + np.cos(tilt) * z_axis
            up_axis_local = np.cos(tilt) * inward - np.sin(tilt) * z_axis
        params = robotics.CameraSensorParams(
            name=name, image_width=int(width), image_height=int(height), fov_vertical_deg=float(fov_vertical_deg),
            near_clip=float(near_clip), far_clip=float(far_clip), offset_local=[float(value) for value in offset_local],
            forward_axis=[float(value) for value in forward_axis],
            up_axis_local=[float(value) for value in up_axis_local], look_distance=0.1)
        # Native SuperDex sensor on the link, destroyed with the bot. parent_from_sensor stays the identity, so the
        # sensor frame is the link frame and the mount lives in the params (as documented for wrist cameras).
        self.sensor = bot.create_sensor("SENSOR_CAMERA", link_name, name, "")
        if self.sensor is None:
            raise RuntimeError(f"bot.create_sensor failed for link {link_name!r}")
        self.sensor.set_params(params)
        self.viewer = viewer
        self.hide = tuple(hide)
        self.restore_transparency_mode = restore_transparency_mode
        self.name = name
        self._depth = None
        self._frustum = None

    def _restore_mode(self):
        """Transparency mode to set back after a wrist render (see ``restore_transparency_mode``)."""
        mode = self.restore_transparency_mode
        if mode != "auto":
            return mode
        structs = list(self.viewer.get_curve_networks()) + list(self.viewer.get_point_clouds())
        return "pretty" if any(struct.get_transparency() < 1.0 for struct in structs) else "none"

    # ---- geometry --------------------------------------------------------------------------------------
    @property
    def params(self):
        """The sensor's CameraSensorParams."""
        return self.sensor.get_params()

    @property
    def size(self):
        """Image (width, height) [px]."""
        params = self.params
        return params.image_width, params.image_height

    @property
    def intrinsics(self) -> np.ndarray:
        """3 x 3 camera matrix in the OpenCV pixel-index convention (see the module docstring)."""
        params = self.params
        width, height = params.image_width, params.image_height
        focal = (height / 2.0) / np.tan(np.deg2rad(params.fov_vertical_deg) / 2.0)
        return np.array([[focal, 0.0, (width - 1) / 2.0], [0.0, focal, (height - 1) / 2.0], [0.0, 0.0, 1.0]])

    def _eye_fwd_up(self):
        """Eye position [m], unit forward direction and image-up direction, in world coordinates."""
        params = self.params
        pose = tf_to_mat(self.sensor.get_world_transform())  # = link pose (parent_from_sensor = identity)
        rotation = pose[:3, :3]
        eye = pose[:3, 3] + rotation @ np.asarray(list(params.offset_local))
        forward = rotation @ np.asarray(list(params.forward_axis))
        forward /= np.linalg.norm(forward)
        up = rotation @ np.asarray(list(params.up_axis_local))
        return eye, forward, up

    def world_from_cam(self) -> np.ndarray:
        """4 x 4 pose of the OpenCV optical frame (x right, y down, z forward) in world coordinates."""
        eye, forward, up = self._eye_fwd_up()
        right = np.cross(forward, up)
        right /= np.linalg.norm(right)
        pose = np.eye(4)
        pose[:3, :3] = np.stack([right, np.cross(forward, right), forward], 1)
        pose[:3, 3] = eye
        return pose

    def project(self, pts_world) -> tuple[np.ndarray, np.ndarray]:
        """Project world points (N, 3) [m] to pixel coordinates (N, 2) (OpenCV index convention) and z-depth (N,)
        [m]."""
        pose = self.world_from_cam()
        cam_points = (np.asarray(pts_world, float) - pose[:3, 3]) @ pose[:3, :3]
        uv = (cam_points @ self.intrinsics.T)[:, :2] / cam_points[:, 2:3]
        return uv, cam_points[:, 2]

    def frustum_nodes(self, depth: float = 0.05) -> np.ndarray:
        """Eye and the 4 image-corner points at z-depth ``depth`` [m] (world), for drawing the camera in external
        views; shape (5, 3)."""
        width, height = self.size
        intrinsics, pose = self.intrinsics, self.world_from_cam()
        corners = np.array([[-0.5, -0.5], [width - 0.5, -0.5], [width - 0.5, height - 0.5], [-0.5, height - 0.5]])
        cam_points = np.c_[(corners[:, 0] - intrinsics[0, 2]) / intrinsics[0, 0],
                           (corners[:, 1] - intrinsics[1, 2]) / intrinsics[1, 1], np.ones(4)] * depth
        return np.vstack([pose[:3, 3], cam_points @ pose[:3, :3].T + pose[:3, 3]])

    def update_frustum(self, depth: float = 0.05, color=(0.9, 0.1, 0.1), radius: float = 0.0012):
        """Draw or refresh this camera's frustum as a curve network "<name>_frustum" (hidden in wrist views).

        Call once per frame before rendering the external views.
        """
        nodes = self.frustum_nodes(depth)
        if self._frustum is None:
            edges = np.array([[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [2, 3], [3, 4], [4, 1]])
            self._frustum = self.viewer.add_curve_network(f"{self.name}_frustum", nodes, edges, radius=radius,
                                                          color=list(color))
        else:
            self._frustum.set_nodes(nodes)

    # ---- rendering ---------------------------------------------------------------------------------------
    def render(self, depth: bool = False, update_scene: bool = True):
        """Render the wrist view.

        The viewer's window size, field of view and length scale are restored afterwards, so external views rendered
        with the same Viewer are unaffected.

        Parameters
        ----------
        depth : bool
            Also return the metric depth image.
        update_scene : bool
            Sync the Viewer's actors first; False skips it (use it when viewer.render() already ran this step).

        Returns
        -------
        rgb : np.ndarray of uint8, shape (H, W, 3)
            The image.
        depth : np.ndarray of float32, shape (H, W)
            z-depth [m], inf where nothing was hit or beyond far_clip (only if ``depth``).
        """
        params = self.params
        width, height = params.image_width, params.image_height
        old_size, old_fov, old_scale = tuple(ps.get_window_size()), ps.get_vertical_fov_degrees(), ps.get_length_scale()
        restore_mode = self._restore_mode()
        hidden = []
        for network in self.viewer.get_curve_networks():
            if network.is_enabled() and any(fnmatch.fnmatchcase(network.get_name(), pattern) for pattern in self.hide):
                network.set_enabled(False)
                hidden.append(network)
        try:
            ps.set_transparency_mode("none")
            if old_size != (width, height):
                ps.set_window_size(width, height)
            ps.set_vertical_fov_degrees(params.fov_vertical_deg)
            ps.set_length_scale(params.near_clip / 0.005)  # Polyscope near = 0.005 L (far = 20 L)
            eye, forward, up = self._eye_fwd_up()
            self.viewer.set_camera_view(look_from=eye, look_at=eye + params.look_distance * forward, up_dir=up)
            img = self.viewer.render() if update_scene else self.viewer.take_screenshot()
            rgb = np.ascontiguousarray(np.asarray(img)[..., :3])
            depth_map = None
            if depth:
                if self._depth is None:
                    self._depth = _GLDepthReader()
                depth_map = self._depth.read_metric(width, height)
                depth_map[depth_map > params.far_clip] = np.inf
        finally:
            if restore_mode is not None:
                ps.set_transparency_mode(restore_mode)
            for network in hidden:
                network.set_enabled(True)
            if old_size != (width, height):
                ps.set_window_size(*old_size)
            ps.set_vertical_fov_degrees(old_fov)
            ps.set_length_scale(old_scale)
        return (rgb, depth_map) if depth else rgb


# ------------------------------------------------------------------------------------------------- helpers
def make_viewer(sim, size=(480, 360), route=None, ssaa: bool = True):
    """SuperDex Viewer for a rope scene: the FR3 GLB visuals and the rope as a tube (the Viewer does not draw rods),
    plus the target route as a thin transparent line.

    Only ONE Viewer can exist per process (Polyscope singleton; Viewer.__init__ raises if Polyscope is initialised).
    ``ssaa=True`` turns on the Viewer's 2x supersampling right away (the Viewer sets ps.set_SSAA_factor(2), but
    Polyscope applies it only at the next window resize, so without this the first frames are aliased and later ones
    are not).

    Returns
    -------
    viewer
        The Viewer.
    sync : callable
        Updates the rope tube; call it before rendering a frame.
    """
    from superdex.physics.viewer.viewer import Viewer
    from superdex.physics.viewer.viewer_cfg import ViewerCfg

    register_visuals(sim)
    viewer = Viewer(ViewerCfg(offscreen=True, size=tuple(size), coordinate_system="ros"))
    viewer.set_scene(sim.scene)
    nodes = sim.rope_nodes()
    rope = viewer.add_curve_network("rope", nodes, np.stack([np.arange(len(nodes) - 1), np.arange(1, len(nodes))], 1),
                                    radius=sim.rp.radius, color=[0.12, 0.38, 0.85])
    if route is not None:
        route_pts = np.c_[route, np.full(len(route), 0.001)]
        viewer.add_curve_network("target", route_pts,
                                 np.stack([np.arange(len(route_pts) - 1), np.arange(1, len(route_pts))], 1),
                                 radius=0.0015, color=[0.55, 0.55, 0.55], transparency=0.6)
    if ssaa:  # the buffers exist only after the first frame; a resize after that re-allocates them at 2x SSAA
        viewer.render()
        ps.set_window_size(size[0] + 1, size[1])
        ps.set_window_size(*size)
    return viewer, (lambda: rope.set_nodes(sim.rope_nodes()))


try:
    import gymnasium as gym

    class WristCameraObs(gym.Wrapper):
        """Adds wrist-camera images to the observations of a rope environment.

        obs = {"state": <original vector>, "rgb": uint8 (H, W, 3)[, "depth": uint8 (H, W, 1)]}; the depth is the
        z-depth mapped linearly from depth_range [m] to 0..255 (uint8, so that SB3 treats it as an image;
        0.5 m / 255 = 2 mm steps). The Viewer is created lazily on the first reset, i.e. inside the SubprocVecEnv
        worker. Use it with SubprocVecEnv (one environment per process); DummyVecEnv with more than one wrapped
        environment fails (one Viewer per process). SB3: PPO("MultiInputPolicy", ...);
        VecNormalize(..., norm_obs_keys=["state"]).
        """

        def __init__(self, env, width: int = 96, height: int = 96, mount: str = "flange_back", depth: bool = False,
                     depth_range=(0.0, 0.5), egl_device: int | None = None, **cam_kwargs):
            super().__init__(env)
            self.width, self.height, self.use_depth, self.depth_range = int(width), int(height), depth, depth_range
            self.cam_kwargs = {**MOUNTS[mount], **cam_kwargs}
            self.egl_device = egl_device
            spaces = {"state": env.observation_space,
                      "rgb": gym.spaces.Box(0, 255, (self.height, self.width, 3), np.uint8)}
            if depth:
                spaces["depth"] = gym.spaces.Box(0, 255, (self.height, self.width, 1), np.uint8)
            self.observation_space = gym.spaces.Dict(spaces)
            self.cam = None

        def _lazy_init(self):
            """Create the Viewer and the camera (in the worker process)."""
            if self.egl_device is not None:
                ps.set_egl_device_index(int(self.egl_device))  # must precede the Polyscope initialisation
            sim = self.env.unwrapped.sim
            self.viewer, self._sync = make_viewer(sim, size=(self.width, self.height))
            self.cam = WristCamera(sim.bot, self.viewer, width=self.width, height=self.height, name="obs_wrist",
                                   restore_transparency_mode=None, **self.cam_kwargs)

        def _obs(self, state):
            """Dictionary observation of the current state."""
            if self.cam is None:
                self._lazy_init()
            self._sync()
            if not self.use_depth:
                return {"state": state, "rgb": self.cam.render()}
            rgb, depth_map = self.cam.render(depth=True)
            low, high = self.depth_range
            scaled = np.clip((np.nan_to_num(depth_map, posinf=high) - low) / (high - low), 0.0, 1.0)
            return {"state": state, "rgb": rgb, "depth": np.round(255 * scaled).astype(np.uint8)[..., None]}

        def reset(self, **kwargs):
            """Reset the wrapped environment and return the dictionary observation."""
            state, info = self.env.reset(**kwargs)
            return self._obs(state), info

        def step(self, action):
            """Step the wrapped environment and return the dictionary observation."""
            state, reward, terminated, truncated, info = self.env.step(action)
            return self._obs(state), reward, terminated, truncated, info

        def close(self):
            """Close the Viewer and the wrapped environment."""
            if self.cam is not None:
                self.viewer.close()
                self.cam = None
            super().close()

except ImportError:  # pragma: no cover
    pass
