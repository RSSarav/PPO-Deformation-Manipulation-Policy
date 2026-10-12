"""HD (1080p, 50 fps) videos of the RAW learned policy, rendered with SuperDex's offscreen viewer.

What the video shows is exactly the policy's rollout, nothing else:

    env.reset(seed=s, options={"layout_seed": s, "no_prefix": True})     (as in verify_rl.py)
    every 0.1 s: a = policy(obs) (deterministic); env.step(a)            until success (terminated) or 50 s

There are no scripted phases, no expert, no release or back-off and no edits. Frames are read from the physics after
every ``--substeps_per_frame`` substeps of 10 ms (default 2, i.e. 50 fps = real time). The capture hook only reads the
state after a substep, so it cannot change the rollout. After the episode ends, its last frame is held for ``--hold``
seconds (no further simulation).

Rendering: Polyscope (headless EGL) at 4x SSAA, glossy FR3 and Robotiq 2F-85 GLB meshes, soft contact shadows,
composited over a lab-bench tabletop and a studio gradient. Each view takes two passes: the tabletop alone, then the
scene on a transparent black background with Polyscope's shadow-only ground plane (premultiplied alpha), composited in
NumPy. The clamp block at the rope's fixed end is a visual prop for the anchor constraint of the physics (node 0
pinned).

The live metrics of the multiview come from ``env._measure()``, the same winding, shape and strain functions that the
reward and the success check use, evaluated at every video frame (read-only). The environment is the one the
checkpoint was trained in: for ``GenericSurfaceEnv`` (notes/B_rl_environment.md) the shape panel shows the
correspondence-free target-surface coverage.

Outputs (data/runs/<run>/hd/<tag>_<view>.mp4, H.264 CRF 16, BT.709, yuv420p, faststart; <tag>.json = outcome):

    multiview  third-person, top and wrist views with live deformation metrics (1920x1080)
    third      full-frame third-person view (the "hero" shot)
    top, wrist full-frame top and wrist-camera views (no overlays)

  cd project_superdex && .venv/bin/python ../rope_routing/scripts/render_hd.py --run generic_ppo1 --model best \
      --seed 9000015
  ... --stills 70   # look development: run 70 policy steps, write one PNG per view, exit

Positions are in metres in the ROS world frame (robot base at the origin); images are uint8 RGB.
``scripts/render_eval_set.py`` and ``scripts/compose_grid.py`` build the evaluation videos from this script, and the
behaviour fingerprint reads :func:`measure`.
"""

import argparse
import json
import os
import subprocess
import sys
import time

import cv2
import imageio_ffmpeg
import numpy as np
import polyscope as ps
import torch
from PIL import Image, ImageDraw, ImageFont

from rope_routing.deformation import shape_distance
from rope_routing.env.generic_surface_env import COVER_SUCCESS
from rope_routing.env.grasp_route_env import PEGS, SUCCESS_SHAPE, SUCCESS_WIND_FRAC
from rope_routing.render import FRANKA_DARK, FRANKA_WHITE, GRIPPER_DARK, GRIPPER_PAD, PEG_WOOD, register_bot_visuals
from rope_routing.sim.wrist_camera import MOUNTS, WristCamera

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
_FONTS = os.path.join(ROOT, "assets", "fonts")  # URW Nimbus Sans (Helvetica metric clone), copied so d3/d4 match
FONT_R, FONT_B = os.path.join(_FONTS, "NimbusSans-Regular.otf"), os.path.join(_FONTS, "NimbusSans-Bold.otf")

# ---------------------------------------------------------------------------------------------------------- look
BENCH = (0.40, 0.41, 0.43)  # lab-bench top
BG_TOP, BG_BOTTOM = (0.86, 0.87, 0.88), (0.62, 0.63, 0.65)  # studio gradient behind the bench
ROPE_BLUE = (0.16, 0.45, 0.86)
CLAMP_GREY = (0.20, 0.20, 0.22)
SHADOW_DARKNESS = 0.55
CAMS = {  # name: (look_from, look_at, up_dir or None, vertical fov [deg]); ROS frame, robot base at the origin
    "third": ([1.45, -1.05, 0.85], [0.40, -0.10, 0.20], None, 35.0),
    "top": ([0.47, -0.10, 2.60], [0.47, -0.10, 0.0], [-1.0, 0.0, 0.0], 15.5),  # near-orthographic, robot at the top
    "close": ([1.12, -0.92, 0.66], [0.44, -0.10, 0.10], None, 35.0),  # tighter third-person framing (grid cells)
}
PEG_NAMES = ("A", "B", "C")
PEG_COLS = [(255, 159, 67), (72, 219, 140), (178, 140, 255)]  # A, B, C (RGB) in the charts and tags


def font(size, bold=False, _cache={}):
    """Nimbus Sans font of the given size [px] (cached)."""
    key = (size, bold)
    if key not in _cache:
        _cache[key] = ImageFont.truetype(FONT_B if bold else FONT_R, size)
    return _cache[key]


def draw_tag(img, x_pos, y_pos, text, size=24, bold=True, fg=(255, 255, 255), bg=(18, 20, 24), alpha=0.72,
             pad=(12, 7), anchor="lt", dot=None):
    """Draw text on a rounded translucent pill, in place on a uint8 RGB image.

    Parameters
    ----------
    img : np.ndarray of uint8, shape (H, W, 3)
        Image, modified in place.
    x_pos, y_pos : float
        Anchor position [px].
    text : str
        Label.
    size : int
        Font size [px].
    bold : bool
        Bold font.
    fg, bg : tuple of int
        Text and pill colours (RGB).
    alpha : float
        Pill opacity.
    pad : tuple of int
        Horizontal and vertical padding [px].
    anchor : str
        Horizontal (l, c, r) and vertical (t, b) anchor of the pill.
    dot : tuple of int, optional
        Colour of a dot before the text.

    Returns
    -------
    tuple of int
        Width and height of the pill [px].
    """
    face = font(size, bold)
    left, top, right, bottom = face.getbbox(text)
    dot_space = (size // 2 + 8) if dot is not None else 0
    width, height = (right - left) + 2 * pad[0] + dot_space, (bottom - top) + 2 * pad[1]
    x0 = int(x_pos - (width if anchor[0] == "r" else width // 2 if anchor[0] == "c" else 0))
    y0 = int(y_pos - (height if anchor[1] == "b" else 0))
    img_h, img_w = img.shape[:2]
    x0, y0 = max(0, min(x0, img_w - width)), max(0, min(y0, img_h - height))
    # Blend the pill into the region, then draw the dot and the text on top.
    region = img[y0:y0 + height, x0:x0 + width].astype(np.float32)
    mask = Image.new("L", (width, height), 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, width - 1, height - 1], radius=height // 2 - 1, fill=255)
    alpha_mask = np.asarray(mask, np.float32)[..., None] / 255.0 * alpha
    region = region * (1 - alpha_mask) + np.asarray(bg, np.float32) * alpha_mask
    pil = Image.fromarray(region.astype(np.uint8))
    draw = ImageDraw.Draw(pil)
    if dot is not None:
        centre_y, dot_r = height // 2, size // 4 + 1
        draw.ellipse([pad[0], centre_y - dot_r, pad[0] + 2 * dot_r, centre_y + dot_r], fill=dot)
    draw.text((pad[0] + dot_space - left, pad[1] - top), text, font=face, fill=fg)
    img[y0:y0 + height, x0:x0 + width] = np.asarray(pil)
    return width, height


# ---------------------------------------------------------------------------------------------------------- video
class Encoder:
    """Raw RGB frames to an H.264 MP4 (BT.709 limited range, yuv420p, faststart).

    GPU: NVENC (h264_nvenc, preset p7 / tune hq, constant quality 16, spatial and temporal AQ, B-frames) with the static
    ffmpeg 8.1 build in data/tools/ffmpeg_nvenc (the bundled imageio ffmpeg has no NVENC; ffmpeg master needs driver >= 610).
    Falls back to CPU libx264 (CRF 16) if NVENC is unavailable.

    Parameters
    ----------
    path : str
        Output file.
    width, height : int
        Frame size [px].
    fps : int
        Frame rate.
    quality : int
        Constant-quality level (NVENC cq / x264 CRF).
    gpu : int, optional
        nvidia-smi index of the GPU used for NVENC.
    """

    NVENC_FFMPEG = os.path.join(ROOT, "data", "tools", "ffmpeg_nvenc", "ffmpeg-n8.1-latest-linux64-gpl-8.1", "bin", "ffmpeg")
    _nvenc_ok = None

    @classmethod
    def nvenc_available(cls):
        """Whether the static ffmpeg can encode with NVENC on this machine (probed once)."""
        if cls._nvenc_ok is None:
            cls._nvenc_ok = False
            if os.path.exists(cls.NVENC_FFMPEG):
                probe = subprocess.run([cls.NVENC_FFMPEG, "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
                                        "color=c=gray:s=256x256:r=25", "-frames:v", "2", "-c:v", "h264_nvenc", "-f",
                                        "null", "-"], capture_output=True)
                cls._nvenc_ok = probe.returncode == 0
        return cls._nvenc_ok

    def __init__(self, path, width, height, fps, quality=16, gpu=None):
        env = None
        if gpu is not None:  # NVENC on the given nvidia-smi GPU (CUDA's device 0 may be another user's full GPU)
            env = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=str(gpu))
        if self.nvenc_available():
            exe = self.NVENC_FFMPEG
            codec = ["-c:v", "h264_nvenc", "-preset", "p7", "-tune", "hq", "-rc", "vbr", "-cq", str(quality), "-b:v",
                     "0", "-maxrate", "80M", "-bufsize", "160M", "-spatial-aq", "1", "-temporal-aq", "1",
                     "-rc-lookahead", "32", "-bf", "3", "-profile:v", "high"]
        else:
            exe = imageio_ffmpeg.get_ffmpeg_exe()
            codec = ["-c:v", "libx264", "-preset", "medium", "-crf", str(quality), "-profile:v", "high", "-threads",
                     "4"]
        cmd = [exe, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r",
               str(fps), "-i", "-", "-an", "-vf", "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p,"
               "setparams=color_primaries=bt709:color_trc=bt709:colorspace=bt709:range=tv", *codec,
               "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709", "-movflags", "+faststart",
               path]
        self.path, self.codec = path, codec[1]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, env=env)

    def write(self, img):
        """Append one RGB frame (H, W, 3)."""
        self.proc.stdin.write(np.ascontiguousarray(img, dtype=np.uint8).tobytes())

    def close(self):
        """Finish the video and wait for ffmpeg."""
        self.proc.stdin.close()
        self.proc.wait()


# ---------------------------------------------------------------------------------------------------------- scene
class HDScene:
    """SuperDex offscreen viewer with two-pass compositing (gradient, tabletop, scene with soft ground shadows).

    Parameters
    ----------
    env : GenericSurfaceEnv or SurfaceRouteEnv
        The environment whose scene is rendered.
    width, height : int
        Frame size [px].
    ssaa : int
        Supersampling factor.
    egl_device : int
        EGL device index; -1 keeps the default (``scripts/egl_probe.py`` maps EGL indices to GPUs).
    """

    def __init__(self, env, width, height, ssaa=4, egl_device=-1):
        from superdex.physics.viewer.renderers.mesh_renderer import MeshRenderer
        from superdex.physics.viewer.viewer import Viewer
        from superdex.physics.viewer.viewer_cfg import ViewerCfg

        self.env, self.width, self.height = env, width, height
        register_bot_visuals(env.sim.scene, env.sim.bot)
        if egl_device >= 0:  # pick a GPU with free memory (scripts/egl_probe.py maps EGL indices to GPUs)
            ps.set_egl_device_index(egl_device)
        self.viewer = Viewer(ViewerCfg(offscreen=True, size=(width, height), coordinate_system="ros"))
        ps.set_SSAA_factor(ssaa)
        self.viewer.set_scene(env.sim.scene)
        self.viewer.render()
        ps.set_window_size(width + 1, height)  # buffers exist only after the first frame: re-allocate them at the SSAA
        ps.set_window_size(width, height)
        ps.set_background_color((0.0, 0.0, 0.0, 0.0))  # transparent black -> premultiplied alpha in captures
        ps.set_ground_plane_height_mode("manual")
        ps.set_ground_plane_height(0.0)
        ps.set_shadow_darkness(SHADOW_DARKNESS)
        ps.set_shadow_blur_iters(5)
        self.ros_to_ps = self.viewer._state.coordinate_transform.source_to_target

        # Materials and colours of the robot and the pegs; the surface meshes are collected for the tabletop pass.
        actors = {state.instance.get_name().split("/")[-1]: state for state in self.viewer._state.scene.actors.values()}
        actors["table"].renderer._render_struct.set_enabled(False)  # SuperDex's infinite checker plane
        self.fg = []  # surface meshes hidden in the tabletop pass (only surfaces: a group toggle shows debug children)
        for name, state in actors.items():
            renderer = state.renderer
            if renderer is None or name == "table" or not hasattr(renderer, "set_front_face_color"):
                continue
            if name.startswith("2f_85"):
                color, material = (GRIPPER_PAD if "finger_tip" in name else GRIPPER_DARK), "clay"
            elif name in ("fr3_link0", "fr3_link8"):
                color, material = FRANKA_DARK, "clay"
            elif name.startswith("fr3_"):
                color, material = FRANKA_WHITE, "ceramic"
            elif name.startswith("peg"):
                color, material = PEG_WOOD, "wax"
            else:
                color, material = None, None
            try:
                if color is not None:
                    renderer.set_front_face_color(color)
                    renderer.set_material(MeshRenderer.Material(material))
            except Exception:
                pass
            if getattr(renderer, "_surface_struct", None) is not None:
                self.fg.append(renderer._surface_struct)

        # Rope, tabletop, anchor clamp, background gradient and the wrist camera.
        nodes = env.sim.rope_nodes()
        edges = np.stack([np.arange(len(nodes) - 1), np.arange(1, len(nodes))], 1)
        self.rope = self.viewer.add_curve_network("rope", nodes, edges, radius=env.sim.rope_radius,
                                                  color=list(ROPE_BLUE))
        self.tab = self._box("tabletop", [0.55, -0.10, -0.025], [2.20, 2.80, 0.05], BENCH, "clay")
        anchor = np.asarray(env.task.anchor, float)
        self.clamp = self._box("clamp", [anchor[0] + 0.012, anchor[1], 0.009], [0.03, 0.03, 0.018], CLAMP_GREY, "clay")
        self.tab.set_enabled(False)
        ps.set_ground_plane_mode("shadow_only")
        ramp = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None, None]
        grad = ((np.asarray(BG_TOP, np.float32) * (1 - ramp) + np.asarray(BG_BOTTOM, np.float32) * ramp)
                * np.ones((1, width, 1), np.float32))
        self.grad = np.ascontiguousarray(np.round(grad * 255.0).astype(np.uint8))
        self.wcam = WristCamera(env.sim.bot, self.viewer, width=width, height=height, restore_transparency_mode="none",
                                **MOUNTS["2f85_side"])
        self._backdrops = {}
        self.peg_px = {}

    def _to_ps(self, points):
        """ROS world points (N, 3) to Polyscope coordinates."""
        points = np.atleast_2d(np.asarray(points, float))
        return (self.ros_to_ps[:3, :3] @ points.T).T + self.ros_to_ps[:3, 3]

    def _box(self, name, centre, size, color, mat):
        """Register an axis-aligned box (centre and size in the ROS frame [m]) as a flat-shaded surface mesh."""
        centre, half = np.asarray(centre, float), np.asarray(size, float) / 2
        vertices = np.array([[sx, sy, sz] for sz in (-1, 1) for sy in (-1, 1) for sx in (-1, 1)], float) * half + centre
        faces = np.array([[0, 2, 3], [0, 3, 1], [4, 5, 7], [4, 7, 6], [0, 1, 5], [0, 5, 4], [2, 6, 7], [2, 7, 3],
                          [0, 4, 6], [0, 6, 2], [1, 3, 7], [1, 7, 5]])
        return ps.register_surface_mesh(name, self._to_ps(vertices), faces, smooth_shade=False, color=list(color),
                                        material=mat)

    @staticmethod
    def _grab():
        """RGBA uint8, premultiplied (transparent black clear colour)."""
        return np.ascontiguousarray(ps.screenshot_to_buffer(transparent_bg=True))

    @staticmethod
    def _over(fg, back):
        """Premultiplied "over" in uint8 with OpenCV SIMD: fg.rgb + back * (1 - fg.a)."""
        inv = cv2.bitwise_not(cv2.extractChannel(fg, 3))
        return cv2.add(cv2.cvtColor(fg, cv2.COLOR_RGBA2RGB),
                       cv2.multiply(back, cv2.merge([inv, inv, inv]), scale=1.0 / 255.0))

    def _backdrop(self):
        """The tabletop pass: the bench alone over the gradient (scene meshes, rope and clamp hidden)."""
        for surface in self.fg:
            surface.set_enabled(False)
        self.rope.set_enabled(False)
        self.clamp.set_enabled(False)
        self.tab.set_enabled(True)
        ps.set_ground_plane_mode("none")
        img = self._grab()
        for surface in self.fg:
            surface.set_enabled(True)
        self.rope.set_enabled(True)
        self.clamp.set_enabled(True)
        self.tab.set_enabled(False)
        ps.set_ground_plane_mode("shadow_only")
        return self._over(img, self.grad)

    def _scene_over(self, back):
        """The scene pass, composited over a backdrop."""
        return self._over(self._grab(), back)

    def sync(self):
        """Update the rope and the actor poses from the physics."""
        self.rope.set_nodes(self.env.sim.rope_nodes())
        self.viewer._state.requires_update = True
        self.viewer._prepare_for_render()

    def _set_cam(self, name):
        """Apply one of the fixed CAMS views."""
        look_from, look_at, up, fov = CAMS[name]
        ps.set_vertical_fov_degrees(fov)
        self.viewer.set_camera_view(look_from=look_from, look_at=look_at, up_dir=up)

    def project(self, points):
        """ROS world points to pixel coordinates in the current camera."""
        cam = self._to_ps(points)
        view = np.asarray(ps.get_camera_view_matrix())
        cam = (view[:3, :3] @ cam.T).T + view[:3, 3]
        focal = 1.0 / np.tan(np.radians(ps.get_vertical_fov_degrees()) / 2)
        x_ndc, y_ndc = focal * (self.height / self.width) * cam[:, 0] / -cam[:, 2], focal * cam[:, 1] / -cam[:, 2]
        return np.c_[(x_ndc + 1) / 2 * self.width, (1 - y_ndc) / 2 * self.height]

    def view(self, name):
        """Render a fixed view ("third", "top" or "close")."""
        self._set_cam(name)
        if name not in self._backdrops:  # static camera: the tabletop pass is rendered once
            self._backdrops[name] = self._backdrop()
            tops = np.c_[PEGS, np.full(3, self.env.task.peg_height + 0.035)]
            self.peg_px[name] = self.project(tops)
        return self._scene_over(self._backdrops[name])

    def wrist(self):
        """Render the wrist-camera view (its backdrop changes with the camera, so both passes run every frame)."""
        params = self.wcam.params
        old_fov, old_scale = ps.get_vertical_fov_degrees(), ps.get_length_scale()
        ps.set_vertical_fov_degrees(params.fov_vertical_deg)
        ps.set_length_scale(params.near_clip / 0.005)  # Polyscope near plane = 0.005 L
        eye, forward, up = self.wcam._eye_fwd_up()
        self.viewer.set_camera_view(look_from=eye, look_at=eye + params.look_distance * forward, up_dir=up)
        img = self._scene_over(self._backdrop())
        ps.set_vertical_fov_degrees(old_fov)
        ps.set_length_scale(old_scale)
        return img

    def close(self):
        """Close the viewer."""
        self.viewer.close()


# ---------------------------------------------------------------------------------------------------------- metrics
def measure(env):
    """Read-only: the environment's own deformation and task measures at the current physics state.

    Returns
    -------
    dict
        "wind" (wrap fraction per peg), "shape" (coverage for GenericSurfaceEnv, ordered shape error otherwise) [m],
        "strain" (largest surface-area stretch, clipped at 0) and "grasped".
    """
    nodes, samples, feats, wind, tcp = env._measure()
    if hasattr(env, "coverage"):  # GenericSurfaceEnv (notes B): wrap fractions per obstacle, target-surface coverage
        wrap, shape = np.asarray(wind, dtype=np.float64), float(env.coverage(feats)[0])
    else:  # SurfaceRouteEnv: raw winding angles, per-sample (ordered) shape error
        wrap = np.clip(wind / env.target_wind, 0.0, 1.0)
        shape = float(shape_distance.ordered(samples, env.sample_goals, env.goal_mask))
    grasped = bool(env._grasp_state()[0])
    # SurfaceRouteEnv returns the fundamental-form descriptor: stretch = surface-area ratio - 1 (notes A)
    strain = feats["area_ratio"] - 1.0 if "area_ratio" in feats else feats["strain"]
    return dict(wind=wrap, shape=shape, strain=max(0.0, float(np.max(strain))), grasped=grasped)  # >= 0: tension


class Strip:
    """Bottom instrument strip of the multiview: live charts of the wrap per peg, the shape error and the peak strain.

    Parameters
    ----------
    width, height : int
        Strip size [px].
    horizon_s : float
        Episode time limit [s] (the charts' time axis).
    left_text, right_text : str
        Header texts.
    generic : bool
        GenericSurfaceEnv labels (wrap relative to the target surface's, correspondence-free coverage).
    """

    BG, FG, DIM, GRID = (20, 22, 26), (236, 238, 242), (150, 156, 168), (52, 56, 64)

    def __init__(self, width, height, horizon_s, left_text, right_text, generic=False):
        self.width, self.height, self.horizon = width, height, horizon_s
        self.right_text = right_text
        img = Image.new("RGB", (width, height), self.BG)
        draw = ImageDraw.Draw(img)
        draw.text((36, 16), left_text, font=font(24, True), fill=self.FG)
        margin, gap, top, bottom = 36, 44, 64, height - 40
        panel_w = (width - 2 * margin - 2 * gap) // 3
        self.panels = []
        # Chart specifications: title, value range, ticks, tick format, success threshold and its label.
        if generic:  # GenericSurfaceEnv: wrap relative to the target surface's wrap; correspondence-free coverage
            shape_spec = ("Distance to the target surface (coverage)", (0.0, 0.15), [0, 0.05, 0.10, 0.15],
                          lambda value: f"{100 * value:.0f} cm", COVER_SUCCESS,
                          f"success < {100 * COVER_SUCCESS:.1f} cm")
        else:
            shape_spec = ("Shape error to the target triangle", (0.0, 0.40), [0, 0.1, 0.2, 0.3, 0.4],
                          lambda value: f"{100 * value:.0f} cm", SUCCESS_SHAPE, "success < 5 cm")
        specs = [("Wrap around each obstacle" if generic else "Wrap around each peg", (0.0, 1.0), [0, 0.5, 1.0],
                  lambda value: f"{100 * value:.0f}%", SUCCESS_WIND_FRAC, "success ≥ 85%"),
                 shape_spec,
                 ("Surface-area stretch (1st fundamental form)", (0.0, 0.03), [0, 0.01, 0.02, 0.03],
                  lambda value: f"{100 * value:.0f}%", 0.02, "penalised > 2%")]
        # Static chart backgrounds: titles, value grid lines, time axis and the dashed success threshold.
        for index, (title, (low, high), ticks, fmt, threshold, thr_label) in enumerate(specs):
            x0 = margin + index * (panel_w + gap)
            px0, px1, py0, py1 = x0 + 64, x0 + panel_w - 8, top + 46, bottom
            draw.text((x0, top), title, font=font(22, True), fill=self.FG)
            for tick in ticks:
                tick_y = py1 - (tick - low) / (high - low) * (py1 - py0)
                draw.line([(px0, tick_y), (px1, tick_y)], fill=self.GRID, width=1)
                label = fmt(tick)
                text_w = draw.textlength(label, font=font(17))
                draw.text((px0 - 10 - text_w, tick_y - 10), label, font=font(17), fill=self.DIM)
            for second in range(0, int(horizon_s) + 1, 10):
                tick_x = px0 + second / horizon_s * (px1 - px0)
                draw.line([(tick_x, py1), (tick_x, py1 + 5)], fill=self.DIM, width=1)
                label = f"{second} s"
                draw.text((tick_x - draw.textlength(label, font=font(16)) / 2, py1 + 8), label, font=font(16),
                          fill=self.DIM)
            thr_y = py1 - (threshold - low) / (high - low) * (py1 - py0)
            for dash_x in range(px0, px1, 14):
                draw.line([(dash_x, thr_y), (min(dash_x + 7, px1), thr_y)], fill=(200, 200, 200), width=2)
            text_w = draw.textlength(thr_label, font=font(16))
            draw.text((px1 - text_w, thr_y - 22), thr_label, font=font(16), fill=(200, 200, 200))
            self.panels.append(dict(x0=x0, px0=px0, px1=px1, py0=py0, py1=py1, lo=low, hi=high, pw=panel_w, top=top))
        self.base = np.asarray(img).copy()
        self.hist = []

    def _pts(self, panel, times, values):
        """Pixel polyline of a chart series, in OpenCV fixed point with 2 fractional bits."""
        x_px = panel["px0"] + np.asarray(times) / self.horizon * (panel["px1"] - panel["px0"])
        y_px = panel["py1"] - ((np.clip(np.asarray(values), panel["lo"], panel["hi"]) - panel["lo"])
                               / (panel["hi"] - panel["lo"]) * (panel["py1"] - panel["py0"]))
        return np.round(np.c_[x_px, y_px] * 4).astype(np.int32)  # 2 fractional bits (cv2 shift=2): smooth AA lines

    def render(self, sim_time, metrics):
        """Append the current metrics (from :func:`measure`) at time ``sim_time`` [s] and draw the strip."""
        self.hist.append((sim_time, *metrics["wind"], metrics["shape"], metrics["strain"]))
        history = np.asarray(self.hist)
        img = self.base.copy()
        # Series: (panel, history column, colour): the three pegs' wraps, the shape error and the peak strain.
        series = [(0, 1, PEG_COLS[0]), (0, 2, PEG_COLS[1]), (0, 3, PEG_COLS[2]), (1, 4, (90, 180, 255)),
                  (2, 5, (255, 110, 150))]
        for panel_index, column, rgb in series:
            points = self._pts(self.panels[panel_index], history[:, 0], history[:, column])
            if len(points) > 1:
                cv2.polylines(img, [points], False, rgb, 3, cv2.LINE_AA, shift=2)
            cv2.circle(img, tuple(points[-1]), 5 * 4, rgb, -1, cv2.LINE_AA, shift=2)
        pil = Image.fromarray(img)
        draw = ImageDraw.Draw(pil)
        panel = self.panels[0]  # current values next to the titles
        text_x = panel["x0"] + panel["pw"]
        for peg in (2, 1, 0):
            text = f"{PEG_NAMES[peg]} {100 * metrics['wind'][peg]:.0f}%"
            text_x -= draw.textlength(text, font=font(20, True)) + (14 if peg < 2 else 0)
            draw.text((text_x, panel["top"] + 2), text, font=font(20, True), fill=PEG_COLS[peg])
        for panel_index, text, rgb in ((1, f"{100 * metrics['shape']:.1f} cm", (90, 180, 255)),
                                       (2, f"{100 * metrics['strain']:.2f}%", (255, 110, 150))):
            panel = self.panels[panel_index]
            draw.text((panel["x0"] + panel["pw"] - draw.textlength(text, font=font(20, True)), panel["top"] + 2), text,
                      font=font(20, True), fill=rgb)
        # Header: grasp indicator and simulated time.
        hold_text = "holding the rope" if metrics["grasped"] else "not holding"
        right = f"{self.right_text}    t = {sim_time:5.2f} s"
        text_w = draw.textlength(right, font=font(22))
        draw.text((self.width - 36 - text_w, 18), right, font=font(22), fill=self.FG)
        hold_w = draw.textlength(hold_text, font=font(22))
        hold_x = self.width - 36 - text_w - 60 - hold_w
        draw.ellipse([hold_x - 24, 22, hold_x - 8, 38], fill=(72, 219, 140) if metrics["grasped"] else (110, 114, 124))
        draw.text((hold_x, 18), hold_text, font=font(22), fill=self.FG if metrics["grasped"] else self.DIM)
        return np.asarray(pil)


# ---------------------------------------------------------------------------------------------------------- main
def load_policy_and_env(args):
    """Load a pure-RL async PPO checkpoint (rl/async_ppo.py) and the environment it was trained in.

    The environment factory and its arguments come from the checkpoint (as in verify_rl.py), so a GenericSurfaceEnv
    policy is rendered in GenericSurfaceEnv and a SurfaceRouteEnv policy in SurfaceRouteEnv. ``--physics tube`` and
    ``--rope`` override the scene configuration.

    Returns
    -------
    env : gym.Env
        The environment (eval mode, seeded with ``args.seed``).
    policy : Policy
        Deterministic mean-action policy.
    """
    from rope_routing.rl.async_ppo import load_policy as load_rl, resolve
    from rope_routing.sim.grasp_scene import SceneCfg

    policy = load_rl(os.path.join(ROOT, "data", "runs", args.run, args.model + ".pt"))
    ck_env = policy.env or {}
    factory = ck_env.get("factory", "rope_routing.env.surface_route_env:make_env")
    kwargs = dict(ck_env.get("kwargs") or {})
    if args.physics == "tube":
        kwargs["scene_cfg"] = SceneCfg.tube_t0()
    if args.rope:  # randomized rope settings (eval_randomized.py): SceneCfg field overrides as a JSON dict
        kwargs["scene_cfg"] = json.loads(args.rope)
    print(f"env: {factory} {sorted(kwargs)} | policy at {policy.global_step:,} env steps", flush=True)
    return resolve(factory)(rank=0, seed=args.seed, eval=True, **kwargs), policy


def main():
    """Render one episode of the policy into the requested videos (or stills) and write <tag>.json."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="surf_ppo2")
    parser.add_argument("--model", default="best_2M")
    parser.add_argument("--kind", choices=["rl"], default="rl")
    parser.add_argument("--physics", choices=["default", "tube"], default="default",
                        help="rl only: SceneCfg() or tube_t0()")
    parser.add_argument("--label", default="learned policy")
    parser.add_argument("--seed", type=int, default=9_000_015)
    parser.add_argument("--outputs", default="multiview,third,top,wrist")
    parser.add_argument("--size", type=int, nargs=2, default=[1920, 1080])
    parser.add_argument("--ssaa", type=int, default=4)
    parser.add_argument("--egl_device", type=int, default=-1, help="EGL device index (d3: 7 = GPU 4, 5 = GPU 6)")
    parser.add_argument("--nvenc_gpu", type=int, default=None, help="nvidia-smi index of the GPU for NVENC encoding")
    parser.add_argument("--substeps_per_frame", type=int, default=2)
    parser.add_argument("--hold", type=float, default=2.0, help="seconds to hold the final frame")
    parser.add_argument("--clean", action="store_true", help="no overlays on the 'third' output (grid cells)")
    parser.add_argument("--third_cam", default="third", choices=["third", "close"],
                        help="framing of the third-person view")
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--tag", default=None)
    parser.add_argument("--max_steps", type=int, default=10_000, help="testing: stop after this many policy steps")
    parser.add_argument("--stills", type=int, default=0, help="look-dev: run N steps, save one PNG per view, exit")
    parser.add_argument("--rope", default=None,
                        help="JSON dict of SceneCfg rope overrides (randomized-rope evaluation)")
    parser.add_argument("--cell_label", default=None, help="label for compose_grid.py cells (stored in <tag>.json)")
    parser.add_argument("--subtitle", default=None, help="right-hand text of the multiview strip")
    args = parser.parse_args()
    torch.set_num_threads(2)
    CAMS["third"] = CAMS[args.third_cam]
    outputs = [output for output in args.outputs.split(",") if output]
    out_dir = args.out_dir or os.path.join(ROOT, "data", "runs", args.run, "hd")
    os.makedirs(out_dir, exist_ok=True)
    tag = args.tag or f"{args.model}_layout{args.seed}"
    width, height = args.size
    fps = 100 // args.substeps_per_frame
    layout_no = args.seed - 9_000_000 if args.seed >= 9_000_000 else args.seed

    # Environment, policy, renderer, chart strip and encoders.
    env, policy = load_policy_and_env(args)
    policy_env = env
    obs, _ = policy_env.reset(seed=args.seed, options={"layout_seed": args.seed, "no_prefix": True})
    if args.stills:
        for _ in range(args.stills):
            obs, reward, terminated, truncated, info = policy_env.step(policy(obs))
            if terminated or truncated:
                break
    scene = HDScene(env, width, height, ssaa=args.ssaa, egl_device=args.egl_device)
    need = set()
    for output in outputs:
        need |= {"multiview": {"third", "top", "wrist"}, "third": {"third"}, "top": {"top"}, "wrist": {"wrist"}}[output]
    strip = (Strip(width, height - height * 2 // 3, env.horizon * 0.1,
                   f"Franka FR3 + Robotiq 2F-85 · rope routing around 3 pegs · {args.label}",
                   args.subtitle or f"unseen layout #{layout_no} · real time", generic=hasattr(env, "coverage"))
             if "multiview" in outputs else None)
    encoders = {} if args.stills else {output: Encoder(os.path.join(out_dir, f"{tag}_{output}.mp4"), width, height, fps,
                                                       gpu=args.nvenc_gpu) for output in outputs}
    if encoders:
        print(f"encoder: {next(iter(encoders.values())).codec}", flush=True)

    def peg_tags(img, name, scale):
        """Label the pegs A, B, C in a view rendered at ``scale`` times the frame size."""
        for peg, (u_px, v_px) in enumerate(scene.peg_px[name] * scale):
            draw_tag(img, u_px, v_px, PEG_NAMES[peg], size=max(14, int(30 * scale)), bold=True, fg=(255, 255, 255),
                     bg=tuple(int(channel * 0.75) for channel in PEG_COLS[peg]), alpha=0.9,
                     pad=(int(9 * scale) + 3, int(5 * scale) + 2), anchor="cb")

    def compose(views, sim_time, metrics, banner=None):
        """Compose the output frames from the rendered views; ``sim_time`` None repeats the last strip (hold)."""
        frames = {}
        if "multiview" in outputs:
            multiview = np.empty((height, width, 3), np.uint8)
            main_w, main_h = width * 2 // 3, height * 2 // 3
            hero = cv2.resize(views["third"], (main_w, main_h), interpolation=cv2.INTER_AREA)
            peg_tags(hero, "third", 2 / 3)
            draw_tag(hero, 16, 16, "Third-person", size=22)
            if banner is not None:
                draw_tag(hero, main_w // 2, 20, banner[0], size=30, bg=banner[1], alpha=0.88, anchor="ct",
                         pad=(22, 12))
            top = cv2.resize(views["top"], (width - main_w, main_h // 2), interpolation=cv2.INTER_AREA)
            peg_tags(top, "top", 1 / 3)
            draw_tag(top, 12, 12, "Top", size=20)
            wrist = cv2.resize(views["wrist"], (width - main_w, main_h - main_h // 2), interpolation=cv2.INTER_AREA)
            draw_tag(wrist, 12, 12, "Wrist camera", size=20)
            multiview[:main_h, :main_w], multiview[:main_h // 2, main_w:], multiview[main_h // 2:main_h, main_w:] = (
                hero, top, wrist)
            multiview[main_h:] = strip.render(sim_time, metrics) if sim_time is not None else strip.last
            if sim_time is not None:
                strip.last = multiview[main_h:].copy()
            multiview[:main_h, main_w - 1:main_w + 1] = 20
            multiview[main_h // 2 - 1:main_h // 2 + 1, main_w:] = 20
            frames["multiview"] = multiview
        if "third" in outputs:
            img = views["third"].copy()
            if not args.clean:
                draw_tag(img, 28, height - 28, f"{args.label} · deterministic · real time", size=26, bold=False,
                         anchor="lb")
                draw_tag(img, width - 28, height - 28,
                         f"t = {(sim_time if sim_time is not None else state['t']):5.2f} s", size=26, bold=False,
                         anchor="rb")
                if banner is not None:
                    draw_tag(img, width // 2, 36, banner[0], size=40, bg=banner[1], alpha=0.88, anchor="ct",
                             pad=(28, 15))
            frames["third"] = img
        for output in ("top", "wrist"):
            if output in outputs:
                frames[output] = views[output]
        return frames

    state = dict(sub=0, t=0.0, frames=0, last=None, m=None)
    prof = dict(measure=0.0, render=0.0, compose=0.0, encode=0.0)

    def capture():
        """Measure, render, compose and encode one frame (read-only on the simulation)."""
        t_start = time.time()
        metrics = measure(env)
        scene.sync()
        t_measured = time.time()
        views = {}
        for name in ("third", "top"):
            if name in need:
                views[name] = scene.view(name)
        if "wrist" in need:
            views["wrist"] = scene.wrist()
        t_rendered = time.time()
        frames = compose(views, state["t"], metrics)
        t_composed = time.time()
        for output, frame in frames.items():
            encoders[output].write(frame)
        t_encoded = time.time()
        for key, duration in zip(prof, (t_measured - t_start, t_rendered - t_measured, t_composed - t_rendered,
                                        t_encoded - t_composed)):
            prof[key] += duration
        state.update(frames=state["frames"] + 1, last=views, m=metrics)

    # Look development: render every view twice (the second round gives steady-state timings) and save PNGs.
    if args.stills:
        metrics = measure(env)
        scene.sync()
        views = {}
        for name in ("third", "top", "wrist", "third", "top", "wrist"):  # second round = steady-state timing
            t_start = time.time()
            views[name] = scene.wrist() if name == "wrist" else scene.view(name)
            print(f"  {name}: {1000 * (time.time() - t_start):.0f} ms", flush=True)
        for name, img in views.items():
            cv2.imwrite(os.path.join(out_dir, f"{tag}_still_{name}.png"), img[..., ::-1])
        if strip is not None:
            still_frames = compose(views, env.elapsed_steps * 0.1, metrics)
            cv2.imwrite(os.path.join(out_dir, f"{tag}_still_multiview.png"), still_frames["multiview"][..., ::-1])
        print("stills ->", out_dir, flush=True)
        scene.close()
        os._exit(0)

    # Episode: the physics substep is wrapped so that a frame is captured every substeps_per_frame substeps.
    orig_substep = env.sim.substep

    def substep_and_capture():
        orig_substep()  # the physics step, untouched
        state["sub"] += 1
        state["t"] = state["sub"] * 0.01
        if state["sub"] % args.substeps_per_frame == 0:
            capture()

    start_time = time.time()
    capture()
    env.sim.substep = substep_and_capture
    info, steps = {}, 0
    while True:
        obs, reward, terminated, truncated, info = policy_env.step(policy(obs))
        steps += 1
        if steps % 10 == 0:
            frames_done = max(state["frames"], 1)
            print(f"  step {steps}  t={state['t']:.1f}s  frames={state['frames']}  "
                  f"{time.time() - start_time:.0f}s wall  per frame: "
                  + "  ".join(f"{key} {1000 * value / frames_done:.0f}ms" for key, value in prof.items()), flush=True)
        if terminated or truncated or steps >= args.max_steps:
            break
    env.sim.substep = orig_substep

    # Hold the final frame with the outcome banner (no simulation), then write the outcome JSON.
    ok = bool(info.get("is_success"))
    wind_frac = np.asarray(info.get("wind_frac", [0, 0, 0]))
    banner = ((f"Solved in {state['t']:.1f} s — rope wrapped around all three pegs", (24, 130, 70)) if ok else
              (f"Not solved ({'time limit' if steps >= env.horizon else 'episode ended'}, {state['t']:.0f} s)",
               (150, 50, 50)))
    for hold_frame in range(int(round(args.hold * fps))):  # hold the final frame (no simulation)
        frames = compose(state["last"], None, state["m"], banner=banner)
        for output, frame in frames.items():
            encoders[output].write(frame)
    for encoder in encoders.values():
        encoder.close()
    result = dict(seed=args.seed, run=args.run, model=args.model, kind=args.kind, success=ok, steps=steps,
                  sim_time=state["t"], shape=float(info.get("shape", np.nan)), wind_frac=wind_frac.tolist(),
                  max_strain=float(info.get("max_strain", np.nan)), frames=state["frames"], fps=fps,
                  wall_s=time.time() - start_time,
                  videos={output: os.path.basename(encoder.path) for output, encoder in encoders.items()},
                  cell_label=args.cell_label, rope=json.loads(args.rope) if args.rope else None)
    json.dump(result, open(os.path.join(out_dir, f"{tag}.json"), "w"), indent=1)
    print(f"{tag}: success={ok} steps={steps} sim={state['t']:.1f}s frames={state['frames']} "
          f"wall={time.time() - start_time:.0f}s -> {out_dir}", flush=True)
    scene.close()
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
