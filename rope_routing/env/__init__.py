"""Gymnasium environments for rope routing (notes/B_rl_environment.md).

``GenericSurfaceEnv`` (generic_surface_env.py) is the final, object-agnostic environment. It derives from
``SurfaceRouteEnv`` (surface_route_env.py), which derives from the first environment, ``GraspRouteEnv``
(grasp_route_env.py). ``start_states`` generates the environment's start-state curriculum.
"""
