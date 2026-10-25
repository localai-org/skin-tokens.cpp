"""
SkinTokens Volumetric Collider Synthesis Package.
Production-grade analytical SDF collider generators and exporters for SkinTokens and skin-tokens.cpp.
"""

from .colliders import (
    ColliderType,
    CoordinateFrame,
    ColliderPrimitive,
    ColliderAssembly,
    IoUResult,
    IoUEvaluator,
    compute_primitive_sdf,
    compute_analytical_sdf,
    synthesize_humanoid_ragdoll,
    synthesize_wardrobe_envelope,
    fit_torus_tangential_capsules,
    export_react_three_rapier_json,
    export_mujoco_mjcf,
    export_ros2_urdf,
)

__all__ = [
    "ColliderType",
    "CoordinateFrame",
    "ColliderPrimitive",
    "ColliderAssembly",
    "IoUResult",
    "IoUEvaluator",
    "compute_primitive_sdf",
    "compute_analytical_sdf",
    "synthesize_humanoid_ragdoll",
    "synthesize_wardrobe_envelope",
    "fit_torus_tangential_capsules",
    "export_react_three_rapier_json",
    "export_mujoco_mjcf",
    "export_ros2_urdf",
]
