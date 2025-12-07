# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import carb
import omni.kit.app
import omni.usd
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, Vt

# Rotational and translational axes used by D6 joints.
ROT_AXES = ("rotX", "rotY", "rotZ")
TRANS_AXES = ("transX", "transY", "transZ")


@dataclass
class RopeParameters:
    """Container describing the cable layout and physical properties."""

    length: float = 1.0  # meters
    radius: float = 0.01  # meters
    segment_count: int = 8
    mass: float = 1.0  # kilograms total
    rot_x_low: float = -30.0  # degrees
    rot_x_high: float = 30.0
    rot_y_low: float = -30.0
    rot_y_high: float = 30.0
    rot_z_low: float = -30.0
    rot_z_high: float = 30.0
    drive_stiffness: float = 1200.0
    drive_damping: float = 70.0
    drive_max_force: float = 200.0
    curve_width_scale: float = 2.0  # multiplier for visual curve width (radius * scale)

    @property
    def segment_length(self) -> float:
        if self.segment_count <= 0:
            return 0.0
        return self.length / float(self.segment_count)

    @property
    def segment_mass(self) -> float:
        if self.segment_count <= 0:
            return 0.0
        return self.mass / float(self.segment_count)

    @property
    def capsule_height(self) -> float:
        """Cylinder height so total capsule length matches the desired segment length."""
        return max(self.segment_length - 2.0 * self.radius, 1e-4)

    @property
    def rot_limits(self) -> Dict[str, Tuple[float, float]]:
        return {
            "rotX": (self.rot_x_low, self.rot_x_high),
            "rotY": (self.rot_y_low, self.rot_y_high),
            "rotZ": (self.rot_z_low, self.rot_z_high),
        }


DEFAULT_PARAMS = RopeParameters()


class RopeBuilderController:
    """Creates a lightweight cable: rigid-body capsules under /World/cable with D6 joints and a spline."""

    def __init__(self):
        self._usd_context = omni.usd.get_context()
        self._params = RopeParameters()
        self._ensure_parameter_defaults()

        self._rope_root_path = "/World/cable"
        self._curve_path = f"{self._rope_root_path}/curve"
        self._rope_exists = False
        self._segment_paths: List[str] = []
        self._joint_paths: List[str] = []
        self._joint_limits: Dict[str, Dict[str, Tuple[float, float]]] = {}
        self._joint_drive_targets: Dict[str, Dict[str, float]] = {}

        self._physics_scene_path = "/World/physicsScene"
        self._update_subscription = None

    @property
    def parameters(self) -> RopeParameters:
        return self._params

    @property
    def curve_path(self) -> str:
        return self._curve_path

    def set_parameters(self, params: RopeParameters):
        self._params = params
        self._ensure_parameter_defaults()
        carb.log_info(f"[RopeBuilder] Updated parameters: {self._params}")

    def create_rope(self) -> str:
        """Create the cable prim hierarchy on the current stage."""
        if not self._validate_params(self._params):
            raise ValueError("Invalid cable parameters. Please fix the highlighted values.")

        stage = self._usd_context.get_stage()
        if stage is None:
            raise RuntimeError("No open USD stage. Create or open a stage before building a cable.")

        self.delete_rope()

        carb.log_info(
            "[RopeBuilder] Building cable with "
            f"{self._params.segment_count} segments, total length {self._params.length} m."
        )

        self._ensure_physics_scene(stage)
        root_prim = UsdGeom.Xform.Define(stage, Sdf.Path(self._rope_root_path))
        root_prim.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.0))

        self._segment_paths = []
        self._joint_paths = []
        self._joint_limits = {}
        self._joint_drive_targets = {}

        prev_segment_path = None
        for idx in range(self._params.segment_count):
            segment_path = self._create_segment(stage, idx)
            if prev_segment_path:
                joint_path = self._create_d6_joint(stage, idx - 1, prev_segment_path, segment_path)
                self._joint_paths.append(joint_path)
            self._segment_paths.append(segment_path)
            prev_segment_path = segment_path

        # Author an initial curve so users see something without subscribing yet.
        self._ensure_curve_prim(stage)
        self._update_curve_points()

        self._rope_exists = True
        return self._rope_root_path

    def delete_rope(self):
        """Remove the cable prims from the stage if they exist."""
        self.stop_curve_updates()

        if not self._rope_exists:
            return

        stage = self._usd_context.get_stage()
        if stage:
            prim = stage.GetPrimAtPath(self._rope_root_path)
            if prim and prim.IsValid():
                stage.RemovePrim(self._rope_root_path)
                carb.log_info("[RopeBuilder] Deleted cable prim hierarchy.")

        self._rope_exists = False
        self._segment_paths = []
        self._joint_paths = []
        self._joint_limits = {}
        self._joint_drive_targets = {}

    def rope_exists(self) -> bool:
        return self._rope_exists

    def validate_parameters(self) -> bool:
        return self._validate_params(self._params)

    def start_curve_updates(self):
        """Subscribe to the per-frame update stream to keep the spline in sync with segment positions."""
        if not self._rope_exists:
            raise RuntimeError("Create a cable before subscribing to the spline update.")

        stage = self._usd_context.get_stage()
        if stage is None:
            raise RuntimeError("No USD stage available.")

        self._ensure_curve_prim(stage)
        if self._update_subscription:
            return

        app = omni.kit.app.get_app()
        stream = app.get_update_event_stream()
        self._update_subscription = stream.create_subscription_to_pop(self._on_curve_update)
        self._update_curve_points()
        carb.log_info("[RopeBuilder] Subscribed to spline updates.")

    def stop_curve_updates(self):
        """Stop per-frame spline updates."""
        if self._update_subscription:
            try:
                self._update_subscription.unsubscribe()
            finally:
                self._update_subscription = None
            carb.log_info("[RopeBuilder] Unsubscribed from spline updates.")

    def curve_subscription_active(self) -> bool:
        return self._update_subscription is not None

    def get_joint_control_data(self) -> List[Dict]:
        """Return joint paths, limits, and current drive targets for UI construction."""
        data = []
        for idx, path in enumerate(self._joint_paths):
            data.append(
                {
                    "index": idx,
                    "path": path,
                    "limits": self._joint_limits.get(path, {}),
                    "targets": self._joint_drive_targets.get(path, {}),
                }
            )
        return data

    def set_joint_drive_target(self, joint_index: int, axis: str, value: float, apply_pose: bool = True) -> float:
        """Clamp a UI-provided target; optionally write to drives or just pose in edit mode."""
        if axis not in ROT_AXES or joint_index < 0 or joint_index >= len(self._joint_paths):
            return 0.0

        joint_path = self._joint_paths[joint_index]
        limits = self._joint_limits.get(joint_path, {})
        low, high = limits.get(axis, (-180.0, 180.0))
        clamped = max(min(value, high), low)

        stage = self._usd_context.get_stage()
        if not stage:
            return clamped

        joint_prim = stage.GetPrimAtPath(joint_path)
        if not joint_prim or not joint_prim.IsValid():
            return clamped

        self._joint_drive_targets.setdefault(joint_path, {})[axis] = clamped

        if apply_pose:
            # Edit-mode shaping: only move the segments, do not modify drive target attributes.
            self._apply_edit_pose_from_targets()
        else:
            # Simulation control path: write drive targets.
            drive = UsdPhysics.DriveAPI.Apply(joint_prim, axis)
            drive.CreateTargetPositionAttr(clamped)
            drive.GetTargetPositionAttr().Set(clamped)
            drive.CreateTargetVelocityAttr(0.0)

        return clamped

    def reset_joint_drive_targets(self):
        """Reset all drive targets to zero within limits."""
        for idx, path in enumerate(self._joint_paths):
            limits = self._joint_limits.get(path, {})
            for axis in ROT_AXES:
                low, high = limits.get(axis, (-180.0, 180.0))
                if low <= 0.0 <= high:
                    self.set_joint_drive_target(idx, axis, 0.0, apply_pose=True)

    def _ensure_parameter_defaults(self):
        """Fill in any missing attributes when hot-reloading older state."""
        for field in DEFAULT_PARAMS.__dataclass_fields__.keys():
            if not hasattr(self._params, field):
                setattr(self._params, field, getattr(DEFAULT_PARAMS, field))

    @staticmethod
    def _validate_params(params: RopeParameters) -> bool:
        """Basic sanity checks to catch common input mistakes."""
        segment_length = params.segment_length
        return all(
            [
                params.length > 0.0,
                params.radius > 0.0,
                params.segment_count > 1,
                params.mass > 0.0,
                segment_length > params.radius * 2.0,
                params.rot_x_low < params.rot_x_high,
                params.rot_y_low < params.rot_y_high,
                params.rot_z_low < params.rot_z_high,
                params.drive_stiffness >= 0.0,
                params.drive_damping >= 0.0,
                params.drive_max_force >= 0.0,
            ]
        )

    def _ensure_physics_scene(self, stage):
        """Create a default PhysX scene if the stage does not have one."""
        scene_prim = stage.GetPrimAtPath(self._physics_scene_path)
        if scene_prim and scene_prim.IsValid():
            return

        scene = UsdPhysics.Scene.Define(stage, Sdf.Path(self._physics_scene_path))
        scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0, 0, -1))
        scene.CreateGravityMagnitudeAttr().Set(9.81)

    def _create_segment(self, stage, index: int) -> str:
        """Author the prims for an individual cable segment (rigid body + collider only)."""
        segment_path = Sdf.Path(f"{self._rope_root_path}/segment_{index:02d}")
        xform = UsdGeom.Xform.Define(stage, segment_path)

        center = self._segment_center_position(index)
        xform.AddTranslateOp().Set(center)

        rigid_prim = xform.GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(rigid_prim)
        mass_api = UsdPhysics.MassAPI.Apply(rigid_prim)
        mass_api.CreateMassAttr(self._params.segment_mass)

        physx_body = PhysxSchema.PhysxRigidBodyAPI.Apply(rigid_prim)
        if physx_body:
            physx_body.CreateEnableCCDAttr(True)

        collision_path = segment_path.AppendPath("collision")
        collision = UsdGeom.Capsule.Define(stage, collision_path)
        collision.CreateRadiusAttr(self._params.radius)
        collision.CreateHeightAttr(self._params.capsule_height)
        collision.CreateAxisAttr(UsdGeom.Tokens.x)
        UsdPhysics.CollisionAPI.Apply(collision.GetPrim())
        UsdGeom.Imageable(collision.GetPrim()).MakeInvisible()

        return str(segment_path)

    def _create_d6_joint(self, stage, joint_index: int, body0_path: str, body1_path: str) -> str:
        """Create a D6 joint connecting two neighboring segments."""
        joint_path = Sdf.Path(f"{self._rope_root_path}/joint_{joint_index:02d}")
        joint_prim = UsdPhysics.Joint.Define(stage, joint_path).GetPrim()
        joint = UsdPhysics.Joint(joint_prim)
        joint.CreateBody0Rel().SetTargets([body0_path])
        joint.CreateBody1Rel().SetTargets([body1_path])

        half_length = self._params.segment_length * 0.5
        joint.CreateLocalPos0Attr().Set(Gf.Vec3f(half_length, 0.0, 0.0))
        joint.CreateLocalPos1Attr().Set(Gf.Vec3f(-half_length, 0.0, 0.0))

        PhysxSchema.PhysxJointAPI.Apply(joint_prim)

        for axis in TRANS_AXES:
            limit = UsdPhysics.LimitAPI.Apply(joint_prim, axis)
            limit.CreateLowAttr(1.0)
            limit.CreateHighAttr(-1.0)

        limits = self._params.rot_limits
        for axis in ROT_AXES:
            low, high = limits[axis]
            limit = UsdPhysics.LimitAPI.Apply(joint_prim, axis)
            limit.CreateLowAttr(low)
            limit.CreateHighAttr(high)

            drive = UsdPhysics.DriveAPI.Apply(joint_prim, axis)
            drive.CreateTypeAttr("force")
            drive.CreateTargetPositionAttr(0.0)
            drive.CreateTargetVelocityAttr(0.0)
            drive.CreateStiffnessAttr(self._params.drive_stiffness)
            drive.CreateDampingAttr(self._params.drive_damping)
            drive.CreateMaxForceAttr(self._params.drive_max_force)

        self._joint_limits[str(joint_path)] = {axis: limits[axis] for axis in ROT_AXES}
        self._joint_drive_targets[str(joint_path)] = {axis: 0.0 for axis in ROT_AXES}
        return str(joint_path)

    def _segment_center_position(self, index: int) -> Gf.Vec3d:
        """Compute the world-space center of the segment assuming a straight cable along X."""
        spacing = self._params.segment_length
        start = -0.5 * self._params.length + 0.5 * spacing
        x = start + index * spacing
        return Gf.Vec3d(x, 0.0, 0.0)

    def _ensure_curve_prim(self, stage):
        curve_prim = UsdGeom.BasisCurves.Get(stage, Sdf.Path(self._curve_path))
        if not curve_prim:
            curve_prim = UsdGeom.BasisCurves.Define(stage, Sdf.Path(self._curve_path)).GetPrim()
            curves = UsdGeom.BasisCurves(curve_prim)
            curves.CreateTypeAttr("cubic")
            curves.CreateBasisAttr("catmullRom")
            curves.CreateWrapAttr("pinned")
        width = max(self._params.radius * self._params.curve_width_scale, 1e-4)
        UsdGeom.BasisCurves(curve_prim).CreateWidthsAttr(Vt.FloatArray([width]))
        return curve_prim

    def _update_curve_points(self):
        stage = self._usd_context.get_stage()
        if not stage or not self._segment_paths:
            return

        curve_prim = self._ensure_curve_prim(stage)
        if not curve_prim:
            return

        pts_list: List[Gf.Vec3f] = []
        for path in self._segment_paths:
            wp = self._segment_world_pos(stage, path)
            if wp is None:
                carb.log_warn(f"[RopeBuilder] Missing segment for curve update: {path}")
                continue
            pts_list.append(wp)

        if len(pts_list) < 2:
            # Not enough points to form a curve; clear counts and points but leave prim.
            curves = UsdGeom.BasisCurves(curve_prim)
            curves.CreateCurveVertexCountsAttr().Set(Vt.IntArray([0]))
            curves.CreatePointsAttr().Set(Vt.Vec3fArray())
            return

        curves = UsdGeom.BasisCurves(curve_prim)
        curves.CreateCurveVertexCountsAttr().Set(Vt.IntArray([len(pts_list)]))
        curves.CreatePointsAttr().Set(Vt.Vec3fArray(pts_list))

    def _on_curve_update(self, _dt):
        if not self._rope_exists:
            self.stop_curve_updates()
            return
        self._update_curve_points()

    def _segment_world_pos(self, stage, path: str) -> Optional[Gf.Vec3f]:
        prim = stage.GetPrimAtPath(path)
        if not prim or not prim.IsValid():
            return None
        xf = UsdGeom.Xformable(prim)
        m = xf.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        pos = m.ExtractTranslation()
        return Gf.Vec3f(pos[0], pos[1], pos[2])

    def _apply_edit_pose_from_targets(self):
        """Reposition segments in edit mode based on current joint drive targets."""
        stage = self._usd_context.get_stage()
        if not stage or not self._segment_paths:
            return

        seg_len = self._params.segment_length
        start_pt = Gf.Vec3d(-0.5 * self._params.length, 0.0, 0.0)
        orientation = Gf.Quatd(1.0, 0.0, 0.0, 0.0)

        for idx, seg_path in enumerate(self._segment_paths):
            if idx > 0 and idx - 1 < len(self._joint_paths):
                joint_path = self._joint_paths[idx - 1]
                targets = self._joint_drive_targets.get(joint_path, {})
                orientation = orientation * self._compose_joint_rotation(targets)

            # Rotate local +X by the accumulated joint orientation.
            forward = Gf.Rotation(orientation).TransformDir(Gf.Vec3d(1.0, 0.0, 0.0))
            end_pt = start_pt + forward * seg_len
            center = (start_pt + end_pt) * 0.5

            prim = stage.GetPrimAtPath(seg_path)
            if prim and prim.IsValid():
                xf = UsdGeom.Xformable(prim)
                xf.ClearXformOpOrder()
                xf.AddTranslateOp().Set(Gf.Vec3f(center))
                qf = Gf.Quatf(float(orientation.GetReal()), Gf.Vec3f(orientation.GetImaginary()))
                xf.AddOrientOp().Set(qf)
                xf.AddScaleOp().Set(Gf.Vec3f(1.0, 1.0, 1.0))

            start_pt = end_pt

        if not self.curve_subscription_active():
            self._update_curve_points()

    def _compose_joint_rotation(self, targets: Dict[str, float]) -> Gf.Quatd:
        """Compose a rotation quaternion from rotX/Y/Z drive targets in degrees."""
        rx = targets.get("rotX", 0.0)
        ry = targets.get("rotY", 0.0)
        rz = targets.get("rotZ", 0.0)

        rot_x = Gf.Rotation(Gf.Vec3d(1.0, 0.0, 0.0), rx)
        rot_y = Gf.Rotation(Gf.Vec3d(0.0, 1.0, 0.0), ry)
        rot_z = Gf.Rotation(Gf.Vec3d(0.0, 0.0, 1.0), rz)

        # Apply rotations in X -> Y -> Z order.
        q = rot_x * rot_y * rot_z
        quatd = q.GetQuat()
        return Gf.Quatd(quatd.GetReal(), quatd.GetImaginary())
