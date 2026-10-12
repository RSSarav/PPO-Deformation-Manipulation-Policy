"""Physics checks of the LIVE GraspScene against the rope-physics lab (notes/C_physics_and_infrastructure.md §1).

The lab's tests were written against vscene.VScene (a frozen copy of the old GraspScene with PhysSpec knobs). Here they
run unmodified against the live scene: ``lab`` swaps vscene's scene class and its rope, solver and contact builders for
the live ones (sim/grasp_scene.py) before executing the lab script, so every number comes from the live code with the
lab's own metric code. The lab's variant names select the live presets:
    V0 -> SceneCfg.legacy_v3()      C1_nocoul -> SceneCfg() (default)      T0_nocoul -> SceneCfg.tube_t0()
and the live preset is first checked field by field against the lab's PhysSpec (a mismatch aborts).

The lab (experiments/rope_physics/exp/scripts, with vscene.py) is not part of this repository; this script needs it
next to rope_routing/ to run.

Usage (from project_superdex/):
  .venv/bin/python ../rope_routing/scripts/check_scene_physics.py parity <out.json> [--n 6] [--steps 150]
      live preset vs VScene variant on the same layouts and actions (rope and TCP trajectories, place error, settle
      length, clearance of the lifted rope from the robot at home and from the pegs)
  .venv/bin/python ../rope_routing/scripts/check_scene_physics.py lab <lab_script.py> <lab args...>
      e.g. lab settle_test.py C1_nocoul out.json --n 25 --start 9000000 ; lab frame_check.py C1_nocoul out.json --n 10
           lab pile_test.py C1_nocoul fold out.json ; lab capstan_test.py C1_nocoul out.json --f_hold 5
           lab speed_test.py out.json V0 C1_nocoul --n 4 ; lab robot_test.py C1_nocoul out.json --n 10
  .venv/bin/python ../rope_routing/scripts/check_scene_physics.py summarize <dir>
      merges the JSON outputs of chunked runs in <dir> into one table (printed)
"""

import json
import os
import runpy
import sys
from dataclasses import replace

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LAB = os.path.join(HERE, "..", "experiments", "rope_physics", "exp", "scripts")
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))
sys.path.insert(0, os.path.abspath(LAB))

import rope_routing.sim.grasp_scene as gs  # noqa: E402

gs._init()
import vscene  # noqa: E402

vscene._initialized = True
LAB_VSCENE = vscene.VScene
LIVE = {"V0": gs.SceneCfg.legacy_v3, "C1_nocoul": gs.SceneCfg, "T0_nocoul": gs.SceneCfg.tube_t0}


def live_cfg(spec):
    """Live preset for a lab variant, with the lab's per-test friction overrides (the capstan test sets table_mu)."""
    if spec.name not in LIVE:
        raise SystemExit(f"no live preset for lab variant {spec.name!r}; known: {sorted(LIVE)}")
    cfg = replace(LIVE[spec.name](), rope_friction=spec.rope_mu, table_friction=spec.table_mu, peg_friction=spec.peg_mu)
    check_cfg_vs_spec(cfg, vscene.SPECS[spec.name])
    return cfg


def check_cfg_vs_spec(cfg, spec):
    """Abort unless every rope-physics knob of the live preset equals the lab variant's."""
    radius = cfg.rope_radius
    rope_pad = radius - gs.DELTA if cfg.rope_pad is None else cfg.rope_pad
    pairs = dict(self_contact=(cfg.rope_self_contact, spec.self_contact), pc_quad=(cfg.rope_pc_quad, spec.pc_quad),
                 pc_radius=(cfg.rope_pc_radius, spec.pc_radius), pc_excl=(cfg.rope_pc_exclusion, spec.pc_excl),
                 rope_pad=(rope_pad, spec.rope_pad), rope_smooth=(cfg.rope_smoothing, spec.rope_smooth),
                 vf=(cfg.rope_friction_falloff, spec.vf), penalty=(cfg.contact_penalty, spec.penalty),
                 tube=(cfg.rope_contact == "tube", spec.tube),
                 line_search=(cfg.line_search, spec.line_search),
                 implicit_n=(cfg.implicit_normal_force, spec.implicit_n),
                 iters=(cfg.nonlinear_iters, spec.iters), nodes=(cfg.rope_nodes, spec.nodes),
                 EI=(cfg.rope_bending_stiffness, spec.EI),
                 GJ=(cfg.rope_torsional_stiffness, spec.GJ), alpha=(cfg.rope_mass_damping, spec.alpha),
                 beta=(cfg.rope_stiffness_damping, spec.beta),
                 joint_coulomb=(cfg.joint_coulomb_friction, spec.joint_coulomb), settle=(cfg.settle, spec.settle),
                 settle_steps=(cfg.settle_steps, spec.settle_steps), settle_v=(cfg.settle_v, spec.settle_v),
                 settle_max=(cfg.settle_max, spec.settle_max),
                 settle_calm=(cfg.settle_calm, spec.settle_calm), z0=(cfg.layout_z0, spec.z0),
                 place=(cfg.placement, {"bc": "bc", "dyn": "frames"}[spec.place]),
                 place_steps=(cfg.place_steps, spec.place_steps),
                 place_z=(cfg.place_z, spec.place_z), static_smooth=(1e-3, spec.static_smooth),
                 rot_inertia=(1.0, spec.rot_inertia_scale),
                 kinetic=(0, spec.kinetic_steps), kappa_cap=(0.0, spec.kappa_cap))
    bad = {key: values for key, values in pairs.items()
           if not (np.isclose(values[0], values[1]) if isinstance(values[0], float) else values[0] == values[1])}
    if bad:
        raise SystemExit(f"live preset != lab {spec.name}: {bad}")


class LiveVScene(gs.GraspScene):
    """The live GraspScene behind the VScene interface that the lab scripts use (spec, status counts, substep hook)."""

    def __init__(self, spec=vscene.PhysSpec(), task=None, cfg=None):
        self.spec, self.n_sub, self.status, self.substep_hook = spec, 0, {"rope": {}, "arm": {}}, None
        super().__init__(task or vscene.make_task(), live_cfg(spec))

    def substep(self):
        """GraspScene.substep, counting the solver convergence status of the rope and the arm and calling the hook."""
        super().substep()
        self.n_sub += 1
        for name, actor in (("rope", self.rope), ("arm", self.arm)):
            status = actor.get_convergence_status().name
            self.status[name][status] = self.status[name].get(status, 0) + 1
        if self.substep_hook is not None:
            self.substep_hook(self)


def _patch_lab():
    """Route vscene's scene class and standalone builders (pile, capstan and braid scenes) to the live code.

    The replacement builders keep vscene's signatures (parameter names included), because the lab scripts call them.
    """
    vscene.VScene = LiveVScene
    vscene.make_solver = lambda sc, spec: gs.configure_solver(sc, live_cfg(spec))

    def make_rope(sc, spec, nodes, name="rope", r=vscene.R_ROPE, rho=0.06, axes=None):
        nodes = np.asarray(nodes, np.float32)
        cfg = replace(live_cfg(spec), rope_radius=r, rope_linear_density=rho)
        return gs.create_rope(sc, cfg, nodes, vscene.bishop(nodes) if axes is None else axes, name)

    def add_table(sc, spec):
        return sc.create_rigid_actor(name="table",
                                     shape=vscene.physics.create_plane_shape(normal=[0, 0, 1], distance=0),
                                     is_static=True, contact=gs.static_contact(live_cfg(spec), spec.table_mu))

    def add_pegs(sc, spec, pegs=vscene.PEGS, peg_radius=0.015, height=0.07):
        vertices, faces = gs.cylinder_mesh(peg_radius, height)
        shape = vscene.physics.create_tri_mesh_shape(vertices.reshape(-1), faces.reshape(-1))
        return [sc.create_rigid_actor(
            name=f"peg{index}", shape=shape, is_static=True, contact=gs.static_contact(live_cfg(spec), spec.peg_mu),
            world_from_local=vscene.physics.TransformRT(translation=[float(peg[0]), float(peg[1]), 0.0]))
            for index, peg in enumerate(pegs)]

    def dyn_place(sc, rope, rest, anchor_c, layout_xy, z_table, steps, z_air, dt, substep=None, k_morph=1e4):
        return gs.frame_consistent_place(sc, rope, rest, anchor_c, layout_xy, z_table, steps, z_air,
                                         substep or (lambda: sc.step(dt)), k_morph)

    vscene.make_rope, vscene.add_table, vscene.add_pegs, vscene.dyn_place = make_rope, add_table, add_pegs, dyn_place


# ====================================================================================================== parity
def parity(out, num_layouts=6, steps=150):
    """Compare the live preset with the lab variant on the same task, layout seeds, reset and actions.

    The actions are environment-style AR(1) random actions through the lab's frozen copy of
    GraspRouteEnv._apply_action, including finger closures and therefore grasps. Writes per-layout rows and the
    worst-case differences to ``out``.
    """
    import robot_test  # lab copy of the env action interface (imports vscene; LAB_VSCENE is still in place)
    task = vscene.make_task()
    results = {}
    for name in ("C1_nocoul", "V0", "T0_nocoul"):
        live = gs.GraspScene(task, live_cfg(vscene.SPECS[name]))
        lab = LAB_VSCENE(vscene.SPECS[name], task)
        rows = []
        for seed in range(9_000_000, 9_000_000 + num_layouts):
            xy = vscene.layouts(lab, [seed])[0]
            clearance = []  # during the placement: rope z range, min distance of rope nodes to robot links / TCP / pads
            orig = live.substep

            def probe():
                orig()
                nodes = live.rope_nodes()
                points = np.r_[live.link_positions(), [live.tcp()], live.pads()]
                if nodes[:, 2].max() > 0.05:  # lifted phase
                    clearance.append((nodes[:, 2].min(), nodes[:, 2].max(),
                                      float(np.linalg.norm(nodes[:, None] - points[None], axis=2).min())))
            live.substep = probe
            live.place(xy)
            live.substep = orig
            lab.place(xy)
            d_place = float(np.abs(live.rope_nodes() - lab.rope_nodes()).max())
            n_live, n_lab = live.settle(), lab.settle()
            d_settle = float(np.abs(live.rope_nodes() - lab.rope_nodes()).max())
            settled_from_layout = float(np.linalg.norm(live.rope_nodes()[:, :2] - xy, axis=1).max())

            # Identical random actions on both scenes; track the largest rope and TCP differences.
            rng = np.random.default_rng(seed)
            act = np.zeros(5)
            max_d_rope, max_d_tcp, grasps = 0.0, 0.0, 0
            for step in range(steps):
                act[:4] = np.clip(0.7 * act[:4] + 0.3 * rng.normal(0, 0.6, 4), -1, 1)
                act[4] = float(np.clip(0.9 * act[4] + 0.1 * rng.normal(0, 3.0), -1, 1))
                command = act.copy()
                if live.tcp()[2] > 0.03:
                    command[2] -= 0.5  # head for the rope so that contacts and grasps happen
                for scene in (live, lab):
                    robot_test.apply_action(scene, command)
                    for _ in range(10):
                        scene.substep()
                max_d_rope = max(max_d_rope, float(np.abs(live.rope_nodes() - lab.rope_nodes()).max()))
                max_d_tcp = max(max_d_tcp, float(np.abs(live.tcp() - lab.tcp()).max()))
                grasps += int(live.grasped)
            clear = np.array(clearance) if clearance else np.zeros((1, 3))
            rows.append(dict(seed=seed, place_err_mm=1000 * live.place_err,
                             place_err_lab_mm=1000 * getattr(lab, "place_err", 0.0),
                             d_rope_after_place=d_place, d_rope_after_settle=d_settle, n_settle_live=n_live,
                             n_settle_lab=n_lab, settled_vs_layout_mm=1000 * settled_from_layout,
                             d_rope_actions_max=max_d_rope, d_tcp_actions_max=max_d_tcp, grasped_steps=grasps,
                             lifted_z_min=float(clear[:, 0].min()), lifted_z_max=float(clear[:, 1].max()),
                             lifted_min_dist_to_robot=float(clear[:, 2].min()), peg_height=live.cfg.peg_height))
            print(name, json.dumps(rows[-1]), flush=True)
        results[name] = dict(rows=rows,
                             max_d_rope=max(max(row["d_rope_after_place"], row["d_rope_after_settle"],
                                                row["d_rope_actions_max"]) for row in rows),
                             max_d_tcp=max(row["d_tcp_actions_max"] for row in rows),
                             settle_equal=all(row["n_settle_live"] == row["n_settle_lab"] for row in rows))
        print("PARITY", name, json.dumps({key: value for key, value in results[name].items() if key != "rows"}),
              flush=True)
    with open(out, "w") as handle:
        json.dump(results, handle, indent=1)


# ====================================================================================================== summarize
def summarize(directory):
    """Merge chunked lab outputs (settle_* and frames_* rows) and print the headline numbers per file group."""
    groups = {}
    for file_name in sorted(os.listdir(directory)):
        if file_name.endswith(".json"):
            groups.setdefault(file_name.split(".")[0].rsplit("_part", 1)[0], []).append(
                os.path.join(directory, file_name))
    for group, files in groups.items():
        data = [json.load(open(path)) for path in files]
        rows = ([row for entry in data for row in entry.get("rows", [])]
                if all("rows" in entry for entry in data) else None)
        if group.startswith("settle") and rows:
            print(group, json.dumps(dict(
                n=len(rows), hop_frac=float(np.mean([row["hop"] for row in rows])),
                lift_idle_max_mm=max(row["lift_idle_mm"] for row in rows),
                creep_median_mm_s=float(np.median([row["creep_last1s_mm_s"] for row in rows])),
                drift_max_mm=max(row["drift_idle_mm"] for row in rows),
                from_layout_median_mm=float(np.median([row["drift_from_layout_mm"] for row in rows])),
                from_layout_max_mm=max(row["drift_from_layout_mm"] for row in rows),
                self_min_mm=min(row["self_min_mm"] for row in rows),
                n_settle_median=float(np.median([row["n_settle"] for row in rows])),
                n_settle_max=max(row["n_settle"] for row in rows),
                t_reset_median_ms=1000 * float(np.median([row["t_reset_s"] for row in rows])),
                substep_ms_median=float(np.median([row["substep_ms"] for row in rows])),
                all_finite=all(row["finite"] for row in rows),
                diverged=sum(row["status"]["rope"].get("DIVERGED", 0) + row["status"]["arm"].get("DIVERGED", 0)
                             for row in rows))))
        elif group.startswith("frames") and rows:
            print(group, json.dumps(dict(
                n=len(rows), adot_t_after_place_max=max(row["adot_t_after_place"] for row in rows),
                adot_t_end_max=max(row["adot_t_end"] for row in rows),
                kink_frac_gt45=float(np.mean([row["bend_max_near_hand_deg"] > 45 for row in rows])),
                bend_near_hand_median=float(np.median([row["bend_max_near_hand_deg"] for row in rows])),
                bend_near_hand_max=max(row["bend_max_near_hand_deg"] for row in rows),
                place_err_max_mm=max(row["place_err_mm"] for row in rows))))
        else:
            for path, entry in zip(files, data):
                summary = entry.get("summary", entry)
                keep = ("dmin_mm", "frac_pass", "frac_pen1mm", "step_ms", "slip_onset_ratio", "mu_eff_measured",
                        "creep_before_onset_mm_s", "finite", "status", "carry", "random", "yaw")
                print(os.path.basename(path), json.dumps(
                    {key: value for key, value in summary.items() if key in keep} if any(key in summary for key in keep)
                    else {key: {inner: inner_value for inner, inner_value in value.items() if inner != "spec"}
                          for key, value in summary.items() if isinstance(value, dict)}))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "parity":
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("out")
        parser.add_argument("--n", type=int, default=6)
        parser.add_argument("--steps", type=int, default=150)
        args = parser.parse_args(sys.argv[2:])
        parity(args.out, args.n, args.steps)
        os._exit(0)
    elif cmd == "lab":
        script = sys.argv[2] if os.path.isabs(sys.argv[2]) else os.path.join(LAB, sys.argv[2])
        _patch_lab()
        sys.argv = [script] + sys.argv[3:]
        runpy.run_path(script, run_name="__main__")
    elif cmd == "summarize":
        summarize(sys.argv[2])
    else:
        print(__doc__)
