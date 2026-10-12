"""Time a 1080p render at a given SSAA factor on one EGL device, to pick an idle GPU for render_hd.py --egl_device.

Arguments: EGL device index, SSAA factor, and optionally the number of seconds to stay alive afterwards (so that
nvidia-smi can show which GPU the EGL device is; on d3, EGL 7 = GPU 4 and EGL 5 = GPU 6).

  cd project_superdex && for i in 0 1 2 3 4 5 6 7; do .venv/bin/python ../rope_routing/scripts/egl_probe.py $i 4; done
"""

import os
import sys
import time

import numpy as np
import polyscope as ps

device, ssaa = int(sys.argv[1]), int(sys.argv[2])
ps.set_egl_device_index(device)
ps.set_allow_headless_backends(True)
ps.set_use_prefs_file(False)
ps.init("openGL3_egl")
ps.set_window_size(1920, 1080)
ps.set_SSAA_factor(ssaa)
# A random mesh of 40000 triangles with a shadow-only ground plane, roughly the cost of a render_hd frame.
vertices = np.random.rand(20000, 3)
faces = np.random.randint(0, 20000, (40000, 3))
ps.register_surface_mesh("m", vertices, faces)
ps.set_ground_plane_mode("shadow_only")
ps.screenshot_to_buffer()
ps.set_window_size(1921, 1080)  # re-allocate the buffers at the SSAA factor
ps.set_window_size(1920, 1080)
ps.screenshot_to_buffer()
start_time = time.time()
for _ in range(5):
    ps.screenshot_to_buffer(transparent_bg=True)
print(f"egl{device} ssaa{ssaa}: {(time.time() - start_time) / 5 * 1000:.0f} ms/frame  pid={os.getpid()}", flush=True)
if len(sys.argv) > 3:  # stay alive so that nvidia-smi can show which GPU this EGL device is
    time.sleep(float(sys.argv[3]))
os._exit(0)
