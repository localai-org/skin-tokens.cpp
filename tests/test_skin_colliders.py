"""
Comprehensive Test Suite for SkinTokens Volumetric Collider Synthesis Package.

Tests:
1. Analytical Signed Distance Functions (Cuboid, Sphere, Capsule Y/Z, Cylinder Y/Z, Composite Union)
2. 64-Primitive Humanoid Ragdoll Synthesizer (anatomical breakdowns, symmetry, parameterization)
3. Wardrobe Flared Capsule Ring Synthesizer (5-tier skirt rings, zero center blockage)
4. Torus & Annular geometry tangential capsule fitting
5. Volumetric IoU evaluation (voxel grid 128^3 and Monte Carlo)
6. Exporters (React Three Rapier JSON, MuJoCo MJCF XML, ROS 2 URDF XML)
7. Coordinate Frame transformations (Three.js Y-up <-> ROS 2 Z-up)
"""

import json
import math
import sys
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import pytest

# Ensure parent directory is in sys.path for direct pytest invocation
_PACKAGE_ROOT = Path(__file__).resolve().parent.parent
if str(_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_ROOT))

from skin_tokens.colliders import (
    ColliderType,
    CoordinateFrame,
    ColliderPrimitive,
    ColliderAssembly,
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


# ============================================================================
# 1. Analytical Signed Distance Function Tests
# ============================================================================

def test_analytical_sdf_cuboid():
    """Test signed distance function for an axis-aligned box/cuboid."""
    box = ColliderPrimitive.create(
        ColliderType.CUBOID,
        position=[0.0, 0.0, 0.0],
        dimensions=[2.0, 2.0, 2.0], # half-extents [1.0, 1.0, 1.0]
        id="box_test",
    )

    pts = np.array([
        [0.0, 0.0, 0.0],       # Center: inside, distance = -1.0
        [1.0, 0.0, 0.0],       # Face center: on surface, distance = 0.0
        [0.0, 1.0, 0.0],       # Top face center: distance = 0.0
        [2.0, 0.0, 0.0],       # Outside +X: distance = 1.0
        [0.0, -3.0, 0.0],      # Outside -Y: distance = 2.0
        [2.0, 2.0, 2.0],       # Corner diagonal: distance = sqrt(1^2 + 1^2 + 1^2) = sqrt(3)
        [0.5, 0.5, 0.5],       # Inside: distance = 0.5 - 1.0 = -0.5
    ], dtype=np.float64)

    sdfs = compute_primitive_sdf(pts, box)
    assert np.isclose(sdfs[0], -1.0, atol=1e-5)
    assert np.isclose(sdfs[1], 0.0, atol=1e-5)
    assert np.isclose(sdfs[2], 0.0, atol=1e-5)
    assert np.isclose(sdfs[3], 1.0, atol=1e-5)
    assert np.isclose(sdfs[4], 2.0, atol=1e-5)
    assert np.isclose(sdfs[5], math.sqrt(3.0), atol=1e-5)
    assert np.isclose(sdfs[6], -0.5, atol=1e-5)


def test_analytical_sdf_sphere():
    """Test signed distance function for a sphere."""
    sphere = ColliderPrimitive.create(
        ColliderType.BALL,
        position=[1.0, 2.0, 3.0],
        radius=1.5,
        id="sphere_test",
    )

    pts = np.array([
        [1.0, 2.0, 3.0],       # Center: distance = -1.5
        [2.5, 2.0, 3.0],       # Surface (+X): distance = 0.0
        [1.0, 3.5, 3.0],       # Surface (+Y): distance = 0.0
        [1.0, 2.0, 4.5],       # Surface (+Z): distance = 0.0
        [4.0, 2.0, 3.0],       # Outside (dist=3.0 from center): distance = 1.5
        [1.0, 2.0, 2.0],       # Inside (dist=1.0 from center): distance = -0.5
    ], dtype=np.float64)

    sdfs = compute_primitive_sdf(pts, sphere)
    assert np.isclose(sdfs[0], -1.5, atol=1e-5)
    assert np.isclose(sdfs[1], 0.0, atol=1e-5)
    assert np.isclose(sdfs[2], 0.0, atol=1e-5)
    assert np.isclose(sdfs[3], 0.0, atol=1e-5)
    assert np.isclose(sdfs[4], 1.5, atol=1e-5)
    assert np.isclose(sdfs[5], -0.5, atol=1e-5)


def test_analytical_sdf_capsule_y_aligned():
    """Test signed distance function for a Y-aligned capsule (Three.js)."""
    capsule = ColliderPrimitive.create(
        ColliderType.CAPSULE,
        position=[0.0, 0.0, 0.0],
        radius=0.5,
        length=2.0, # trunk height = 2.0 - 2*0.5 = 1.0 (from y=-0.5 to y=+0.5)
        id="cap_y",
        frame=CoordinateFrame.THREE_JS,
    )

    pts = np.array([
        [0.0, 0.0, 0.0],       # Center: distance = -0.5
        [0.5, 0.0, 0.0],       # Radial surface: distance = 0.0
        [0.0, 1.0, 0.0],       # Top spherical cap apex (y=0.5+0.5=1.0): distance = 0.0
        [0.0, -1.0, 0.0],      # Bottom spherical cap apex: distance = 0.0
        [0.0, 2.0, 0.0],       # Above top apex: distance = 1.0
        [1.5, 0.0, 0.0],       # Outside radial: distance = 1.0
    ], dtype=np.float64)

    sdfs = compute_primitive_sdf(pts, capsule)
    assert np.isclose(sdfs[0], -0.5, atol=1e-5)
    assert np.isclose(sdfs[1], 0.0, atol=1e-5)
    assert np.isclose(sdfs[2], 0.0, atol=1e-5)
    assert np.isclose(sdfs[3], 0.0, atol=1e-5)
    assert np.isclose(sdfs[4], 1.0, atol=1e-5)
    assert np.isclose(sdfs[5], 1.0, atol=1e-5)


def test_analytical_sdf_capsule_z_aligned():
    """Test signed distance function for a Z-aligned capsule (ROS 2 REP-103)."""
    capsule = ColliderPrimitive.create(
        ColliderType.CAPSULE,
        position=[0.0, 0.0, 0.0],
        radius=0.5,
        length=2.0, # trunk from z=-0.5 to z=+0.5
        id="cap_z",
        frame=CoordinateFrame.ROS2_REP103,
    )

    pts = np.array([
        [0.0, 0.0, 0.0],       # Center: distance = -0.5
        [0.5, 0.0, 0.0],       # Radial surface: distance = 0.0
        [0.0, 0.0, 1.0],       # Top cap apex (z=1.0): distance = 0.0
        [0.0, 0.0, -1.0],      # Bottom cap apex (z=-1.0): distance = 0.0
        [0.0, 0.0, 2.0],       # Outside Z: distance = 1.0
    ], dtype=np.float64)

    sdfs = compute_primitive_sdf(pts, capsule)
    assert np.isclose(sdfs[0], -0.5, atol=1e-5)
    assert np.isclose(sdfs[1], 0.0, atol=1e-5)
    assert np.isclose(sdfs[2], 0.0, atol=1e-5)
    assert np.isclose(sdfs[3], 0.0, atol=1e-5)
    assert np.isclose(sdfs[4], 1.0, atol=1e-5)


def test_analytical_sdf_cylinder():
    """Test signed distance function for a cylinder."""
    cylinder = ColliderPrimitive.create(
        ColliderType.CYLINDER,
        position=[0.0, 0.0, 0.0],
        radius=1.0,
        length=2.0, # y in [-1.0, 1.0]
        id="cyl_test",
        frame=CoordinateFrame.THREE_JS,
    )

    pts = np.array([
        [0.0, 0.0, 0.0],       # Center: distance = -min(1.0, 1.0) = -1.0
        [1.0, 0.0, 0.0],       # Radial side surface: distance = 0.0
        [0.0, 1.0, 0.0],       # Top flat cap: distance = 0.0
        [0.0, -1.0, 0.0],      # Bottom flat cap: distance = 0.0
        [2.0, 0.0, 0.0],       # Outside side: distance = 1.0
        [0.0, 3.0, 0.0],       # Outside top: distance = 2.0
        [2.0, 2.0, 0.0],       # Outside corner: distance = sqrt((2-1)^2 + (2-1)^2) = sqrt(2)
    ], dtype=np.float64)

    sdfs = compute_primitive_sdf(pts, cylinder)
    assert np.isclose(sdfs[0], -1.0, atol=1e-5)
    assert np.isclose(sdfs[1], 0.0, atol=1e-5)
    assert np.isclose(sdfs[2], 0.0, atol=1e-5)
    assert np.isclose(sdfs[3], 0.0, atol=1e-5)
    assert np.isclose(sdfs[4], 1.0, atol=1e-5)
    assert np.isclose(sdfs[5], 2.0, atol=1e-5)
    assert np.isclose(sdfs[6], math.sqrt(2.0), atol=1e-5)


def test_composite_union_sdf():
    """Test composite union SDF: SDF_union(p) = min_i SDF_i(p)."""
    s1 = ColliderPrimitive.create(ColliderType.BALL, [0.0, 0.0, 0.0], radius=1.0, id="s1")
    s2 = ColliderPrimitive.create(ColliderType.BALL, [5.0, 0.0, 0.0], radius=1.0, id="s2")

    assembly = ColliderAssembly(primitives=[s1, s2])

    pts = np.array([
        [0.0, 0.0, 0.0],  # Center of s1: distance = -1.0
        [5.0, 0.0, 0.0],  # Center of s2: distance = -1.0
        [2.5, 0.0, 0.0],  # Midpoint: distance to s1 is 1.5, to s2 is 1.5 -> min is 1.5
    ], dtype=np.float64)

    sdfs = compute_analytical_sdf(pts, assembly)
    assert np.isclose(sdfs[0], -1.0, atol=1e-5)
    assert np.isclose(sdfs[1], -1.0, atol=1e-5)
    assert np.isclose(sdfs[2], 1.5, atol=1e-5)


# ============================================================================
# 2. 64-Primitive Humanoid Ragdoll Synthesizer Tests
# ============================================================================

def test_64_primitive_humanoid_ragdoll_generation():
    """Test 64-primitive humanoid ragdoll synthesizer parameterization, breakdown, and bilateral symmetry."""
    min_bounds = [-0.4, 0.0, -0.15]
    max_bounds = [0.4, 1.7, 0.15]

    ragdoll = synthesize_humanoid_ragdoll(min_bounds, max_bounds, max_primitives=64)
    prims = ragdoll.primitives

    # Total primitives must not exceed 64
    assert len(prims) <= 64
    assert len(prims) >= 55  # Full parameterization has ~60 primitives

    # Verify anatomical breakdown by IDs
    ids = [p.id for p in prims]

    # 1. Head & Facial (10)
    head_ids = [i for i in ids if "head" in i or "cranium" in i or "forehead" in i or "jaw" in i or "chin" in i or "nose" in i or "cheek" in i or "ear" in i]
    assert len(head_ids) == 10

    # 2. Neck & Trapezius / Clavicles (6)
    neck_ids = [i for i in ids if "neck" in i or "trapezius" in i or "clavicle" in i]
    assert len(neck_ids) == 6

    # 3. Torso & Core (8)
    torso_ids = [i for i in ids if "chest" in i or "spine" in i or "hips" in i or "hip_" in i]
    assert len(torso_ids) == 8

    # 4. Bilateral Bust (4)
    bust_ids = [i for i in ids if "bust" in i]
    assert len(bust_ids) == 4

    # 5. Bilateral Glutes (4)
    glute_ids = [i for i in ids if "glute" in i]
    assert len(glute_ids) == 4

    # 6. Upper/Lower Arms & Hands (8)
    arm_ids = [i for i in ids if "arm" in i or "elbow" in i or "hand" in i]
    assert len(arm_ids) == 8

    # 7. Thighs, Patella, Calves, Feet (8)
    leg_ids = [i for i in ids if "leg" in i or "knee" in i or "foot" in i]
    assert len(leg_ids) == 8

    # 8. Individual Digits (10)
    digit_ids = [i for i in ids if "thumb" in i or "index" in i or "ring" in i or "distal" in i]
    assert len(digit_ids) == 10

    # Test Bilateral Symmetry (Left vs Right)
    left_cheek = next(p for p in prims if p.id == "left_cheek_sphere")
    right_cheek = next(p for p in prims if p.id == "right_cheek_sphere")
    assert np.isclose(left_cheek.position[0], -right_cheek.position[0], atol=1e-5)
    assert np.isclose(left_cheek.position[1], right_cheek.position[1], atol=1e-5)
    assert np.isclose(left_cheek.radius, right_cheek.radius, atol=1e-5)

    left_bust = next(p for p in prims if p.id == "left_bust_sphere")
    right_bust = next(p for p in prims if p.id == "right_bust_sphere")
    assert np.isclose(left_bust.position[0], -right_bust.position[0], atol=1e-5)
    assert np.isclose(left_bust.position[1], right_bust.position[1], atol=1e-5)
    assert np.isclose(left_bust.radius, right_bust.radius, atol=1e-5)

    left_glute = next(p for p in prims if p.id == "left_glute_sphere")
    right_glute = next(p for p in prims if p.id == "right_glute_sphere")
    assert np.isclose(left_glute.position[0], -right_glute.position[0], atol=1e-5)
    assert np.isclose(left_glute.position[1], right_glute.position[1], atol=1e-5)


# ============================================================================
# 3. Wardrobe Flared Capsule Ring Synthesizer & Torus Tests
# ============================================================================

def test_wardrobe_envelope_synthesis():
    """Test wardrobe flared capsule ring synthesizer and zero center hole blockage."""
    min_bounds = [-0.5, 0.0, -0.2]
    max_bounds = [0.5, 1.6, 0.2]

    wardrobe = synthesize_wardrobe_envelope(min_bounds, max_bounds, max_primitives=64)
    assert len(wardrobe.primitives) <= 64

    # Verify presence of skirt rings, jacket, sleeves, boots
    ids = [p.id for p in wardrobe.primitives]
    assert any("waist_ring" in i for i in ids)
    assert any("upper_skirt_ring" in i for i in ids)
    assert any("mid_skirt_ring" in i for i in ids)
    assert any("lower_skirt_ring" in i for i in ids)
    assert any("hemline_ring" in i for i in ids)
    assert any("jacket_chest" in i for i in ids)
    assert any("sleeve" in i for i in ids)
    assert any("boot" in i for i in ids)

    # Test ZERO Center Hole Blockage:
    # Query points down the central Y-axis through the skirt envelope
    # (x=0, y in skirt heights, z=0). The SDF MUST be strictly positive (> 0),
    # verifying that the center hole of the skirt/dress is completely open and unblocked.
    H = 1.6
    skirt_y_levels = [0.18 * H, 0.26 * H, 0.34 * H, 0.42 * H, 0.50 * H]
    center_query_pts = np.array([[0.0, y, 0.0] for y in skirt_y_levels], dtype=np.float64)

    # Filter primitives to only skirt ring capsules
    ring_prims = [p for p in wardrobe.primitives if "ring" in p.id]
    ring_assembly = ColliderAssembly(primitives=ring_prims)

    ring_sdfs = compute_analytical_sdf(center_query_pts, ring_assembly)
    for idx, sdf_val in enumerate(ring_sdfs):
        # Must be strictly positive (hollow inside)
        assert sdf_val > 0.01, f"Center hole blockage detected at skirt level y={skirt_y_levels[idx]}: SDF={sdf_val}"


def test_torus_tangential_capsules_fitting():
    """Test toroidal/annular ring decomposition into tangential capsules."""
    center = [0.0, 1.0, 0.0]
    major_r = 0.5
    minor_r = 0.1
    num_caps = 8

    torus = fit_torus_tangential_capsules(
        center=center,
        major_radius=major_r,
        minor_radius=minor_r,
        num_capsules=num_caps,
        normal_axis=1, # Y-normal, ring in XZ plane
    )

    assert len(torus.primitives) == 8

    # Query center of the torus ring: should be hollow (SDF > 0)
    center_pt = np.array([[0.0, 1.0, 0.0]], dtype=np.float64)
    center_sdf = compute_analytical_sdf(center_pt, torus)
    assert center_sdf[0] > 0.1, f"Torus center is blocked: SDF={center_sdf[0]}"

    # Query points on the major ring circumference: should be inside or on surface (SDF <= 0)
    ring_pts = np.array([
        [0.5, 1.0, 0.0],
        [-0.5, 1.0, 0.0],
        [0.0, 1.0, 0.5],
        [0.0, 1.0, -0.5],
    ], dtype=np.float64)

    ring_sdfs = compute_analytical_sdf(ring_pts, torus)
    for idx, s in enumerate(ring_sdfs):
        assert s <= 1e-4, f"Ring point {idx} is not inside collider envelope: SDF={s}"


# ============================================================================
# 4. Volumetric IoU Evaluator & Self-Consistency Tests
# ============================================================================

def test_volumetric_iou_self_consistency():
    """Test that evaluating a collider assembly against itself yields IoU = 1.0 (100%)."""
    min_bounds = [-0.3, 0.0, -0.1]
    max_bounds = [0.3, 1.5, 0.1]
    ragdoll = synthesize_humanoid_ragdoll(min_bounds, max_bounds, max_primitives=64)

    # 1. Test 128^3 Voxel Grid evaluation
    res_voxel = IoUEvaluator.evaluate_voxel_grid(
        target_colliders=ragdoll,
        predicted_colliders=ragdoll,
        grid_resolution=64, # 64^3 for fast unit test
    )

    assert res_voxel.iou >= 0.998  # Exact self IoU = 1.0 (>= 99.8%)
    assert res_voxel.precision >= 0.998
    assert res_voxel.recall >= 0.998
    assert res_voxel.passed_threshold is True
    assert res_voxel.parsimony > 0.0

    # 2. Test Stratified Monte Carlo evaluation
    res_mc = IoUEvaluator.evaluate_monte_carlo(
        target_colliders=ragdoll,
        predicted_colliders=ragdoll,
        num_samples=50000,
        seed=42,
    )

    assert res_mc.iou >= 0.998
    assert res_mc.precision >= 0.998
    assert res_mc.recall >= 0.998
    assert res_mc.passed_threshold is True


def test_volumetric_iou_parsimony_and_bounds():
    """Test that disjoint shapes yield IoU = 0.0 and overlapping shapes yield expected IoU."""
    b1 = ColliderPrimitive.create(ColliderType.CUBOID, [0.0, 0.0, 0.0], dimensions=[1.0, 1.0, 1.0])
    b2 = ColliderPrimitive.create(ColliderType.CUBOID, [10.0, 10.0, 10.0], dimensions=[1.0, 1.0, 1.0])

    res = IoUEvaluator.evaluate_voxel_grid([b1], [b2], grid_resolution=32)
    assert res.iou == 0.0
    assert res.passed_threshold is False


# ============================================================================
# 5. Direct Exporters Tests (Rapier JSON, MuJoCo MJCF, ROS 2 URDF)
# ============================================================================

def test_exporters_rapier_json_validation():
    """Test React Three Rapier JSON exporter format and schema validity."""
    min_bounds = [-0.3, 0.0, -0.1]
    max_bounds = [0.3, 1.5, 0.1]
    ragdoll = synthesize_humanoid_ragdoll(min_bounds, max_bounds, max_primitives=64)

    json_str = export_react_three_rapier_json(ragdoll)
    data = json.loads(json_str)

    assert isinstance(data, list)
    assert len(data) == len(ragdoll.primitives)

    for item in data:
        assert "id" in item
        assert "type" in item
        assert item["type"] in ["CuboidCollider", "BallCollider", "CapsuleCollider", "CylinderCollider"]
        assert "args" in item
        assert isinstance(item["args"], list)
        assert len(item["args"]) >= 1

        if item["type"] == "CuboidCollider":
            assert len(item["args"]) == 3
        elif item["type"] == "BallCollider":
            assert len(item["args"]) == 1
        elif item["type"] in ["CapsuleCollider", "CylinderCollider"]:
            assert len(item["args"]) == 2

        assert "position" in item and len(item["position"]) == 3
        assert "rotation" in item and len(item["rotation"]) == 3
        assert "quaternion" in item and len(item["quaternion"]) == 4
        assert "density" in item
        assert "sensor" in item


def test_exporters_mujoco_mjcf_validation():
    """Test MuJoCo MJCF XML exporter format and parseability."""
    min_bounds = [-0.3, 0.0, -0.1]
    max_bounds = [0.3, 1.5, 0.1]
    ragdoll = synthesize_humanoid_ragdoll(min_bounds, max_bounds, max_primitives=64)

    xml_str = export_mujoco_mjcf(ragdoll, body_name="test_avatar")
    root = ET.fromstring(xml_str)

    assert root.tag == "mujoco"
    wb = root.find("worldbody")
    assert wb is not None
    body = wb.find("body")
    assert body is not None
    assert body.attrib.get("name") == "test_avatar"

    geoms = body.findall("geom")
    assert len(geoms) == len(ragdoll.primitives)

    for geom in geoms:
        assert "name" in geom.attrib
        assert "type" in geom.attrib
        assert geom.attrib["type"] in ["box", "sphere", "capsule", "cylinder"]
        assert "size" in geom.attrib
        assert "pos" in geom.attrib
        assert "quat" in geom.attrib


def test_exporters_ros2_urdf_validation():
    """Test ROS 2 URDF XML exporter format and parseability."""
    min_bounds = [-0.3, 0.0, -0.1]
    max_bounds = [0.3, 1.5, 0.1]
    ragdoll = synthesize_humanoid_ragdoll(min_bounds, max_bounds, max_primitives=64)

    xml_str = export_ros2_urdf(ragdoll, link_name="torso_link", robot_name="gemma_ragdoll")
    root = ET.fromstring(xml_str)

    assert root.tag == "robot"
    assert root.attrib.get("name") == "gemma_ragdoll"
    link = root.find("link")
    assert link is not None
    assert link.attrib.get("name") == "torso_link"

    collisions = link.findall("collision")
    assert len(collisions) == len(ragdoll.primitives)

    for col in collisions:
        origin = col.find("origin")
        assert origin is not None
        assert "xyz" in origin.attrib
        assert "rpy" in origin.attrib

        geom = col.find("geometry")
        assert geom is not None
        assert len(geom) == 1
        child_tag = geom[0].tag
        assert child_tag in ["box", "sphere", "capsule", "cylinder"]


# ============================================================================
# 6. Coordinate Frame Transformation Tests
# ============================================================================

def test_coordinate_frame_transformations():
    """Test roundtrip transformations between Three.js (Y-up) and ROS 2 (Z-up)."""
    p_three = ColliderPrimitive.create(
        ColliderType.CUBOID,
        position=[1.0, 2.0, 3.0],
        dimensions=[0.2, 0.4, 0.6],
        rotation_euler=(0.1, 0.2, 0.3),
        id="frame_test",
        frame=CoordinateFrame.THREE_JS,
    )

    p_ros = p_three.to_ros_frame()
    assert p_ros.frame == CoordinateFrame.ROS2_REP103

    p_roundtrip = p_ros.to_three_frame()
    assert p_roundtrip.frame == CoordinateFrame.THREE_JS
    assert np.allclose(p_roundtrip.position, p_three.position, atol=1e-5)
    assert np.allclose(p_roundtrip.dimensions, p_three.dimensions, atol=1e-5)


if __name__ == "__main__":
    tests = [
        test_analytical_sdf_cuboid,
        test_analytical_sdf_sphere,
        test_analytical_sdf_capsule_y_aligned,
        test_analytical_sdf_capsule_z_aligned,
        test_analytical_sdf_cylinder,
        test_composite_union_sdf,
        test_64_primitive_humanoid_ragdoll_generation,
        test_wardrobe_envelope_synthesis,
        test_torus_tangential_capsules_fitting,
        test_volumetric_iou_self_consistency,
        test_volumetric_iou_parsimony_and_bounds,
        test_exporters_rapier_json_validation,
        test_exporters_mujoco_mjcf_validation,
        test_exporters_ros2_urdf_validation,
        test_coordinate_frame_transformations,
    ]

    print(f"Executing {len(tests)} SkinTokens test cases...")
    passed = 0
    for t in tests:
        try:
            t()
            print(f"  [PASS] {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {t.__name__}: {e}")

    print(f"\nResults: {passed}/{len(tests)} passed.")
    if passed != len(tests):
        sys.exit(1)

