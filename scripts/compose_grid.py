"""Grid video: the same raw policy on several episodes side by side, in real time (default 4 x 3 = 12 cells).

The cells come from render_hd.py (--outputs third --size 960 540 --third_cam close --clean --hold 0 --tag cell_<seed>),
i.e. <cells_dir>/cell_<seed>_third.mp4 and cell_<seed>.json. When a cell's episode ends, its last frame is held with a
badge (solved in X s / not solved); the title bar counts the solved episodes live.

  python ../rope_routing/scripts/compose_grid.py --cells_dir data/runs/<run>/eval_randomized_rope/cells \
      --seeds 9100003 9100007 ... --title "12 successes" --out data/runs/<run>/eval_randomized_rope/successes_12.mp4
"""

import argparse
import importlib.util
import json
import os

import cv2
import imageio_ffmpeg
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
_spec = importlib.util.spec_from_file_location("render_hd", os.path.join(ROOT, "scripts", "render_hd.py"))
rhd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rhd)

FRAME_W, FRAME_H, BAR = 1920, 1080, 60  # output frame size and title-bar height [px]
GREEN, RED = (24, 130, 70), (150, 50, 50)


def cell_frames(path, cell_w, cell_h):
    """Decoded RGB frames of one cell video, cropped to the cell aspect ratio and resized to cell_w x cell_h."""
    frames = imageio_ffmpeg.read_frames(path)
    meta = frames.__next__()
    video_w, video_h = meta["size"]
    crop = int(round(video_w * cell_h / cell_w))
    if crop <= video_h:
        y0, x0, crop_w = (video_h - crop) // 2, 0, video_w
    else:  # wider cell than the video: crop the width instead
        crop, crop_w = video_h, int(round(video_h * cell_w / cell_h))
        y0, x0 = 0, (video_w - crop_w) // 2
    for buffer in frames:
        frame = np.frombuffer(buffer, np.uint8).reshape(video_h, video_w, 3)[y0:y0 + crop, x0:x0 + crop_w]
        yield cv2.resize(frame, (cell_w, cell_h), interpolation=cv2.INTER_AREA)


def main():
    """Compose the grid video from the cell videos and write it with render_hd's encoder."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--cells_dir", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--cols", type=int, default=4)
    parser.add_argument("--rows", type=int, default=3)
    parser.add_argument("--title", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--hold", type=float, default=3.0)
    parser.add_argument("--fps", type=int, default=50)
    parser.add_argument("--nvenc_gpu", type=int, default=None)
    args = parser.parse_args()
    assert len(args.seeds) <= args.cols * args.rows, "more seeds than cells"
    cell_w, cell_h = FRAME_W // args.cols, (FRAME_H - BAR) // args.rows
    cells = []
    for seed in args.seeds:
        result = json.load(open(os.path.join(args.cells_dir, f"cell_{seed}.json")))
        cells.append(dict(seed=seed, res=result,
                          it=cell_frames(os.path.join(args.cells_dir, f"cell_{seed}_third.mp4"), cell_w, cell_h),
                          last=None, done=False))
    n_total = max(cell["res"]["frames"] for cell in cells) + int(args.hold * args.fps)
    encoder = rhd.Encoder(args.out, FRAME_W, FRAME_H, args.fps, gpu=args.nvenc_gpu)
    n_ok = sum(cell["res"]["success"] for cell in cells)
    for frame_index in range(n_total):
        sim_time = frame_index / args.fps
        canvas = np.full((FRAME_H, FRAME_W, 3), 20, np.uint8)
        solved_now = 0
        # Cells: the next frame of each episode, or its last frame with the outcome badge once it has ended.
        for index, cell in enumerate(cells):
            if not cell["done"]:
                try:
                    cell["last"] = next(cell["it"])
                except StopIteration:
                    cell["done"] = True
            img = cell["last"].copy()
            row, col = divmod(index, args.cols)
            rhd.draw_tag(img, 8, 8, cell["res"].get("cell_label", f"episode {cell['seed']}"), size=17, bold=True,
                         pad=(9, 4))
            if cell["done"]:
                ok = cell["res"]["success"]
                solved_now += ok
                cv2.rectangle(img, (1, 1), (cell_w - 2, cell_h - 2), GREEN if ok else RED, 3)
                badge = f"✓ solved in {cell['res']['sim_time']:.1f} s" if ok else "✗ not solved"
                rhd.draw_tag(img, cell_w // 2, cell_h - 10, badge, size=20, bold=True, bg=GREEN if ok else RED,
                             alpha=0.9, anchor="cb", pad=(12, 5))
            top, left = BAR + row * cell_h, col * cell_w
            canvas[top:top + cell_h, left:left + cell_w] = img
        # Grid lines and the title bar.
        for row in range(1, args.rows):
            canvas[BAR + row * cell_h - 1:BAR + row * cell_h + 1] = 20
        for column in range(1, args.cols):
            canvas[BAR:, column * cell_w - 1:column * cell_w + 1] = 20
        rhd.draw_tag(canvas, 20, 12, f"{args.title}  ·  real time", size=24, bold=True, bg=(20, 20, 20), alpha=0.0)
        rhd.draw_tag(canvas, FRAME_W - 20, 12,
                     f"t = {min(sim_time, n_total / args.fps):5.1f} s    solved: {solved_now} / {len(cells)}",
                     size=24, bold=True, bg=(20, 20, 20), alpha=0.0, anchor="rt")
        encoder.write(canvas)
    encoder.close()
    print(f"{args.out}: {n_total} frames, {n_ok}/{len(cells)} solved", flush=True)


if __name__ == "__main__":
    main()
