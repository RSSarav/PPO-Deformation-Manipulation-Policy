"""Pure-RL rope routing: an FR3 arm with a Robotiq 2F-85 gripper routes an anchored rope around three pegs in SuperDex.

Subpackages: ``sim`` (scene, physics settings, wrist camera), ``task`` (peg layout, routing path, random rope
layouts), ``deformation`` (surface-area and curvature measures of the deformation), ``env`` (gymnasium environments)
and ``rl`` (asynchronous PPO). ``render`` and ``viz`` hold the 3-D and matplotlib rendering helpers.
"""
