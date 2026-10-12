"""Top-down and side-view matplotlib plots and GIFs of the rope, the pegs, the target route and the FR3 (headless).

Positions are in metres in the robot base frame. This module predates the 3-D renderers (``render.py`` and
``scripts/render_hd.py``) and is not used by the training or evaluation code.
"""

import matplotlib

matplotlib.use("Agg")
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np


def _top(ax, task, nodes, samples=None, color_by=None, title="", tcp=None, arm=None):
    """Draw the top view (x-y): target route, pegs A, B, C, arm links, rope, samples, anchor and gripper."""
    ax.clear()
    ax.plot(task.route[:, 0], task.route[:, 1], "--", color="0.6", lw=1, label="target route")
    for index, peg in enumerate(task.pegs):
        ax.add_patch(plt.Circle(peg, task.peg_radius, color="0.25"))
        ax.text(peg[0] + 0.02, peg[1] + 0.02, "ABC"[index], fontsize=10)
    if arm is not None:
        ax.plot(arm[:, 0], arm[:, 1], "-o", color="tab:gray", lw=4, ms=3, alpha=0.6, label="FR3 (links)")
    ax.plot(nodes[:, 0], nodes[:, 1], "-", color="tab:blue", lw=2.5, label="rope")
    if samples is not None:
        if color_by is not None:
            ax.scatter(samples[:, 0], samples[:, 1], c=color_by, cmap="viridis", s=18, zorder=3)
        else:
            ax.scatter(samples[:, 0], samples[:, 1], s=12, color="k", zorder=3)
    ax.plot(*task.anchor, "s", color="tab:red", ms=7, label="anchor")
    if tcp is not None:
        ax.plot(tcp[0], tcp[1], "o", color="tab:orange", ms=8, label="gripper")
    ax.set_aspect("equal")
    ax.set_xlim(-0.25, 0.85)
    ax.set_ylim(-0.75, 0.3)
    ax.set_title(title, fontsize=9)
    ax.legend(loc="lower right", fontsize=7)


def _side(ax, task, nodes, tcp=None, arm=None):
    """Draw the side view (x-z): pegs, table, arm links, rope and gripper."""
    ax.clear()
    for peg in task.pegs:
        ax.add_patch(plt.Rectangle((peg[0] - task.peg_radius, 0), 2 * task.peg_radius, task.peg_height,
                                   color="0.25"))
    ax.axhline(0, color="0.5", lw=1)
    if arm is not None:
        ax.plot(arm[:, 0], arm[:, 2], "-o", color="tab:gray", lw=4, ms=3, alpha=0.6)
    ax.plot(nodes[:, 0], nodes[:, 2], "-", color="tab:blue", lw=2)
    if tcp is not None:
        ax.plot(tcp[0], tcp[2], "o", color="tab:orange", ms=8)
    ax.set_xlim(-0.25, 0.85)
    ax.set_ylim(-0.02, 1.0)
    ax.set_aspect("equal")
    ax.set_title("side view (x-z)", fontsize=9)


def save_png(path, task, nodes, tcp=None, arm=None, **top_kwargs):
    """Save a top and side view of one state as a PNG.

    Parameters
    ----------
    path : str
        Output file.
    task : TriangleTask
        Pegs, anchor and route.
    nodes : np.ndarray, shape (N, 3)
        Rope nodes [m].
    tcp : np.ndarray, shape (3,), optional
        Gripper position [m].
    arm : np.ndarray, shape (n_links, 3), optional
        Link positions of the arm [m].
    **top_kwargs
        "samples", "color_by" and "title" of the top view.
    """
    fig, (top_ax, side_ax) = plt.subplots(1, 2, figsize=(9, 4.8), gridspec_kw={"width_ratios": [1, 1]})
    _top(top_ax, task, nodes, tcp=tcp, arm=arm, **top_kwargs)
    _side(side_ax, task, nodes, tcp=tcp, arm=arm)
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def save_gif(path, task, frames, titles=None, tcps=None, arms=None, every=1, fps=10):
    """Save an animated GIF of top and side views, one frame per ``every``-th entry of ``frames``.

    Parameters
    ----------
    path : str
        Output file.
    task : TriangleTask
        Pegs, anchor and route.
    frames : sequence of np.ndarray, shape (N, 3)
        Rope nodes per frame [m].
    titles, tcps, arms : sequence, optional
        Per-frame titles, gripper positions and arm link positions.
    every : int
        Frame stride.
    fps : float
        Playback rate [frames/s].
    """
    fig, (top_ax, side_ax) = plt.subplots(1, 2, figsize=(9, 4.8), gridspec_kw={"width_ratios": [1, 1]})
    images = []
    for index in range(0, len(frames), every):
        tcp = None if tcps is None else tcps[index]
        arm = None if arms is None else arms[index]
        _top(top_ax, task, frames[index], title=titles[index] if titles else f"step {index}", tcp=tcp, arm=arm)
        _side(side_ax, task, frames[index], tcp=tcp, arm=arm)
        fig.canvas.draw()
        images.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
    plt.close(fig)
    imageio.mimsave(path, images, duration=1.0 / fps, loop=0)
