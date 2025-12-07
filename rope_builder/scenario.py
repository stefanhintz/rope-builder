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

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

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


@dataclass
class CableState:
    root_path: str
    curve_path: str
    params: RopeParameters
    segment_paths: List[str]
    joint_paths: List[str]
    joint_limits: Dict[str, Dict[str, Tuple[float, float]]]
    joint_drive_targets: Dict[str, Dict[str, float]]
    show_curve: bool = True
    update_subscription: Optional[Any] = None


DEFAULT_PARAMS = RopeParameters()


class RopeBuilderController:
    """Creates and manages multiple lightweight cables with D6 joints and splines."""

    def __init__(self):
        self._usd_context = omni.usd.get_context()
        self._params = RopeParameters()
        self._ensure_parameter_defaults()

        self._physics_scene_path = "/World/physicsScene"
        self._cables: Dict[str, CableState] = {}
        self._active_path: Optional[str] = None

    @property
    def parameters(self) -> RopeParameters:
        return self._params

    def set_parameters(self, params: RopeParameters):
        self._params = params
        self._ensure_parameter_defaults()
        carb.log_info(f"[RopeBuilder] Updated parameters: {self._params}")

    def active_cable_path(self) -> Optional[str]:
        return self._active_path

    def rope_exists(self) -> bool:
        return self._active_path in self._cables

    def list_cable_paths(self) -> List[str]:
        return sorted(self._cables.keys())

    def set_active_cable(self, root_path: str) -> bool:
        if root_path in self._cables:
            self._active_path = root_path
            return True
        return False

    def create_rope(self, name: Optional[str] = None) -> str:
        """Create a new cable prim hierarchy on the current stage."""
        if not self._validate_params(self._params):
            raise ValueError("Invalid cable parameters. Please fix the highlighted values.")

        stage = self._require_stage()
        self._ensure_physics_scene(stage)

        root_path = self._make_unique_root_path(stage, name or "cable")
        params = RopeParameters(**vars(self._params))

        segment_paths: List[str] = []
        joint_paths: List[str] = []
        joint_limits: Dict[str, Dict[str, Tuple[float, float]]] = {}
        joint_targets: Dict[str, Dict[str, float]] = {}

        root_prim = UsdGeom.Xform.Define(stage, Sdf.Path(root_path))
        root_prim.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.0))

        prev_segment_path = None
        for idx in range(params.segment_count):
            segment_path = self._create_segment(stage, root_path, params, idx)
            if prev_segment_path:
                joint_path, limits = self._create_d6_joint(stage, root_path, params, idx - 1, prev_segment_path, segment_path)
                joint_paths.append(joint_path)
                joint_limits[joint_path] = limits
                joint_targets[joint_path] = {axis: 0.0 for axis in ROT_AXES}
            segment_paths.append(segment_path)
            prev_segment_path = segment_path

        curve_path = f"{root_path}/curve"
        state = CableState(
            root_path=root_path,
            curve_path=curve_path,
            params=params,
            segment_paths=segment_paths,
            joint_paths=joint_paths,
            joint_limits=joint_limits,
            joint_drive_targets=joint_targets,
        )
        self._cables[root_path] = state
        self._active_path = root_path

        self._ensure_curve_prim(stage, state)
        self._update_curve_points(state)
        self._apply_visibility_state(state)

        carb.log_info(
            "[RopeBuilder] Built cable %s with %d segments, length %.3f m."
            % (root_path, params.segment_count, params.length)
        )
        return root_path

    def import_cable(self, root_path: str) -> str:
        """Add an existing cable under root_path to the tool and set it active."""
        stage = self._require_stage()
        if root_path in self._cables:
            self._active_path = root_path
            return root_path

        root_prim = stage.GetPrimAtPath(root_path)
        if not root_prim or not root_prim.IsValid():
            raise RuntimeError(f"Root prim not found: {root_path}")

        seg_regex = re.compile(r"segment_(\d+)$")
        segments = []
        for child in root_prim.GetChildren():
            m = seg_regex.search(child.GetName())
            if m:
                segments.append((int(m.group(1)), child.GetPath().pathString))
        segments.sort(key=lambda x: x[0])
        segment_paths = [p for _, p in segments]
        if len(segment_paths) < 2:
            raise RuntimeError("Need at least 2 segments to import a cable.")

        joint_regex = re.compile(r"joint_(\d+)")
        joints = []
        for child in root_prim.GetChildren():
            m = joint_regex.search(child.GetName())
            if m:
                joints.append((int(m.group(1)), child.GetPath().pathString))
        joints.sort(key=lambda x: x[0])
        joint_paths = [p for _, p in joints]

        params, joint_limits = self._infer_params_from_stage(stage, root_path, segment_paths, joint_paths)
        joint_targets = {jp: {axis: 0.0 for axis in ROT_AXES} for jp in joint_paths}

        curve_path = f"{root_path}/curve"
        state = CableState(
            root_path=root_path,
            curve_path=curve_path,
            params=params,
            segment_paths=segment_paths,
            joint_paths=joint_paths,
            joint_limits=joint_limits,
            joint_drive_targets=joint_targets,
        )
        self._cables[root_path] = state
        self._active_path = root_path

        self._ensure_curve_prim(stage, state)
        self._update_curve_points(state)
        self._apply_visibility_state(state)
        carb.log_info(f"[RopeBuilder] Imported cable at {root_path}.")
        return root_path

    def delete_rope(self, root_path: Optional[str] = None):
        """Remove a cable from the stage and controller."""
        state = self._get_state(root_path, require=False)
        if not state:
            return

        self.stop_curve_updates(state.root_path)

        stage = self._usd_context.get_stage()
        if stage:
            prim = stage.GetPrimAtPath(state.root_path)
            if prim and prim.IsValid():
                stage.RemovePrim(state.root_path)
                carb.log_info(f"[RopeBuilder] Deleted cable prim hierarchy at {state.root_path}.")

        self._cables.pop(state.root_path, None)
        if self._active_path == state.root_path:
            self._active_path = self.list_cable_paths()[0] if self._cables else None

    def validate_parameters(self) -> bool:
        return self._validate_params(self._params)

    def start_curve_updates(self, root_path: Optional[str] = None):
        """Subscribe to per-frame updates for a cable spline."""
        state = self._get_state(root_path)
        stage = self._require_stage()

        self._ensure_curve_prim(stage, state)
        if state.update_subscription:
            return

        app = omni.kit.app.get_app()
        stream = app.get_update_event_stream()
        state.update_subscription = stream.create_subscription_to_pop(
            lambda dt, rp=state.root_path: self._on_curve_update(rp, dt)
        )
        self._update_curve_points(state)
        carb.log_info(f"[RopeBuilder] Subscribed spline updates for {state.root_path}.")

    def stop_curve_updates(self, root_path: Optional[str] = None):
        """Stop per-frame spline updates for a cable."""
        state = self._get_state(root_path, require=False)
        if state and state.update_subscription:
            try:
                state.update_subscription.unsubscribe()
            finally:
                state.update_subscription = None
            carb.log_info(f"[RopeBuilder] Unsubscribed spline updates for {state.root_path}.")

    def curve_subscription_active(self) -> bool:
        state = self._get_state(require=False)
        return bool(state and state.update_subscription)

    def toggle_visibility(self) -> bool:
        """Toggle between showing spline or collision capsules on the active cable."""
        state = self._get_state()
        state.show_curve = not state.show_curve
        self._apply_visibility_state(state)
        return state.show_curve

    def showing_curve(self) -> bool:
        state = self._get_state(require=False)
        return True if state is None else state.show_curve

    def get_joint_control_data(self) -> List[Dict]:
        state = self._get_state(require=False)
        if not state:
            return []
        data = []
        for idx, path in enumerate(state.joint_paths):
            data.append(
                {
                    "index": idx,
                    "path": path,
                    "limits": state.joint_limits.get(path, {}),
                    "targets": state.joint_drive_targets.get(path, {}),
                }
            )
        return data

    def set_joint_drive_target(
        self, joint_index: int, axis: str, value: float, apply_pose: bool = True
    ) -> float:
        """Clamp a UI-provided target; optionally write to drives or just pose in edit mode."""
        state = self._get_state()

        if axis not in ROT_AXES or joint_index < 0 or joint_index >= len(state.joint_paths):
            return 0.0

        joint_path = state.joint_paths[joint_index]
        limits = state.joint_limits.get(joint_path, {})
        low, high = limits.get(axis, (-180.0, 180.0))
        clamped = max(min(value, high), low)

        stage = self._usd_context.get_stage()
        if not stage:
            return clamped

        state.joint_drive_targets.setdefault(joint_path, {})[axis] = clamped

        if apply_pose:
            # Edit-mode shaping: only move the segments, do not modify drive target attributes.
            self._apply_edit_pose_from_targets(state)
        else:
            # Simulation control path: write drive targets.
            joint_prim = stage.GetPrimAtPath(joint_path)
            if joint_prim and joint_prim.IsValid():
                drive = UsdPhysics.DriveAPI.Apply(joint_prim, axis)
                drive.CreateTargetPositionAttr(clamped)
                drive.GetTargetPositionAttr().Set(clamped)
                drive.CreateTargetVelocityAttr(0.0)

        return clamped

    def reset_joint_drive_targets(self):
        """Reset all drive targets to zero within limits for the active cable."""
        state = self._get_state(require=False)
        if not state:
            return
        for idx, path in enumerate(state.joint_paths):
            limits = state.joint_limits.get(path, {})
            for axis in ROT_AXES:
                low, high = limits.get(axis, (-180.0, 180.0))
                if low <= 0.0 <= high:
                    self.set_joint_drive_target(idx, axis, 0.0, apply_pose=True)

    # ----------------------------------------------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------------------------------------------

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

    def _create_segment(self, stage, root_path: str, params: RopeParameters, index: int) -> str:
        """Author the prims for an individual cable segment (rigid body + collider only)."""
        segment_path = Sdf.Path(f"{root_path}/segment_{index:02d}")
        xform = UsdGeom.Xform.Define(stage, segment_path)

        center = self._segment_center_position(params, index)
        xform.AddTranslateOp().Set(center)

        rigid_prim = xform.GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(rigid_prim)
        mass_api = UsdPhysics.MassAPI.Apply(rigid_prim)
        mass_api.CreateMassAttr(params.segment_mass)

        physx_body = PhysxSchema.PhysxRigidBodyAPI.Apply(rigid_prim)
        if physx_body:
            physx_body.CreateEnableCCDAttr(True)

        collision_path = segment_path.AppendPath("collision")
        collision = UsdGeom.Capsule.Define(stage, collision_path)
        collision.CreateRadiusAttr(params.radius)
        collision.CreateHeightAttr(params.capsule_height)
        collision.CreateAxisAttr(UsdGeom.Tokens.x)
        UsdPhysics.CollisionAPI.Apply(collision.GetPrim())
        UsdGeom.Imageable(collision.GetPrim()).MakeInvisible()

        return str(segment_path)

    def _create_d6_joint(
        self,
        stage,
        root_path: str,
        params: RopeParameters,
        joint_index: int,
        body0_path: str,
        body1_path: str,
    ) -> Tuple[str, Dict[str, Tuple[float, float]]]:
        """Create a D6 joint connecting two neighboring segments."""
        joint_path = Sdf.Path(f"{root_path}/joint_{joint_index:02d}")
        joint_prim = UsdPhysics.Joint.Define(stage, joint_path).GetPrim()
        joint = UsdPhysics.Joint(joint_prim)
        joint.CreateBody0Rel().SetTargets([body0_path])
        joint.CreateBody1Rel().SetTargets([body1_path])

        half_length = params.segment_length * 0.5
        joint.CreateLocalPos0Attr().Set(Gf.Vec3f(half_length, 0.0, 0.0))
        joint.CreateLocalPos1Attr().Set(Gf.Vec3f(-half_length, 0.0, 0.0))

        PhysxSchema.PhysxJointAPI.Apply(joint_prim)

        for axis in TRANS_AXES:
            limit = UsdPhysics.LimitAPI.Apply(joint_prim, axis)
            limit.CreateLowAttr(1.0)
            limit.CreateHighAttr(-1.0)

        limits = params.rot_limits
        limits_out: Dict[str, Tuple[float, float]] = {}
        for axis in ROT_AXES:
            low, high = limits[axis]
            limit = UsdPhysics.LimitAPI.Apply(joint_prim, axis)
            limit.CreateLowAttr(low)
            limit.CreateHighAttr(high)

            drive = UsdPhysics.DriveAPI.Apply(joint_prim, axis)
            drive.CreateTypeAttr("force")
            drive.CreateTargetPositionAttr(0.0)
            drive.CreateTargetVelocityAttr(0.0)
            drive.CreateStiffnessAttr(params.drive_stiffness)
            drive.CreateDampingAttr(params.drive_damping)
            drive.CreateMaxForceAttr(params.drive_max_force)

            limits_out[axis] = (low, high)

        return str(joint_path), limits_out

    def _segment_center_position(self, params: RopeParameters, index: int) -> Gf.Vec3d:
        """Compute the world-space center of the segment assuming a straight cable along X."""
        spacing = params.segment_length
        start = -0.5 * params.length + 0.5 * spacing
        x = start + index * spacing
        return Gf.Vec3d(x, 0.0, 0.0)

    def _ensure_curve_prim(self, stage, state: CableState):
        curve_prim = UsdGeom.BasisCurves.Get(stage, Sdf.Path(state.curve_path))
        if not curve_prim:
            curve_prim = UsdGeom.BasisCurves.Define(stage, Sdf.Path(state.curve_path)).GetPrim()
            curves = UsdGeom.BasisCurves(curve_prim)
            curves.CreateTypeAttr("cubic")
            curves.CreateBasisAttr("catmullRom")
            curves.CreateWrapAttr("pinned")
        width = max(state.params.radius * state.params.curve_width_scale, 1e-4)
        UsdGeom.BasisCurves(curve_prim).CreateWidthsAttr(Vt.FloatArray([width]))
        return curve_prim

    def _update_curve_points(self, state: CableState):
        stage = self._usd_context.get_stage()
        if not stage or not state.segment_paths:
            return

        curve_prim = self._ensure_curve_prim(stage, state)
        if not curve_prim:
            return

        pts_list: List[Gf.Vec3f] = []
        for path in state.segment_paths:
            wp = self._segment_world_pos(stage, path)
            if wp is None:
                carb.log_warn(f"[RopeBuilder] Missing segment for curve update: {path}")
                continue
            pts_list.append(wp)

        curves = UsdGeom.BasisCurves(curve_prim)
        if len(pts_list) < 2:
            curves.CreateCurveVertexCountsAttr().Set(Vt.IntArray([0]))
            curves.CreatePointsAttr().Set(Vt.Vec3fArray())
            return

        curves.CreateCurveVertexCountsAttr().Set(Vt.IntArray([len(pts_list)]))
        curves.CreatePointsAttr().Set(Vt.Vec3fArray(pts_list))

    def _on_curve_update(self, root_path: str, _dt):
        state = self._cables.get(root_path)
        if not state:
            return
        self._update_curve_points(state)

    def _segment_world_pos(self, stage, path: str) -> Optional[Gf.Vec3f]:
        prim = stage.GetPrimAtPath(path)
        if not prim or not prim.IsValid():
            return None
        xf = UsdGeom.Xformable(prim)
        m = xf.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        pos = m.ExtractTranslation()
        return Gf.Vec3f(pos[0], pos[1], pos[2])

    def _apply_edit_pose_from_targets(self, state: CableState):
        """Reposition segments in edit mode based on current joint drive targets."""
        stage = self._usd_context.get_stage()
        if not stage or not state.segment_paths:
            return

        params = state.params
        seg_len = params.segment_length
        start_pt = Gf.Vec3d(-0.5 * params.length, 0.0, 0.0)
        orientation = Gf.Quatd(1.0, 0.0, 0.0, 0.0)

        for idx, seg_path in enumerate(state.segment_paths):
            if idx > 0 and idx - 1 < len(state.joint_paths):
                joint_path = state.joint_paths[idx - 1]
                targets = state.joint_drive_targets.get(joint_path, {})
                orientation = orientation * self._compose_joint_rotation(targets)

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

        if not state.update_subscription:
            self._update_curve_points(state)

    def _compose_joint_rotation(self, targets: Dict[str, float]) -> Gf.Quatd:
        """Compose a rotation quaternion from rotX/Y/Z drive targets in degrees."""
        rx = targets.get("rotX", 0.0)
        ry = targets.get("rotY", 0.0)
        rz = targets.get("rotZ", 0.0)

        rot_x = Gf.Rotation(Gf.Vec3d(1.0, 0.0, 0.0), rx)
        rot_y = Gf.Rotation(Gf.Vec3d(0.0, 1.0, 0.0), ry)
        rot_z = Gf.Rotation(Gf.Vec3d(0.0, 0.0, 1.0), rz)

        q = rot_x * rot_y * rot_z
        quatd = q.GetQuat()
        return Gf.Quatd(quatd.GetReal(), quatd.GetImaginary())

    def _apply_visibility_state(self, state: CableState):
        """Show either the spline or the collision capsules to declutter the view."""
        stage = self._usd_context.get_stage()
        if not stage:
            return

        show_curve = state.show_curve

        curve_prim = stage.GetPrimAtPath(state.curve_path)
        if curve_prim and curve_prim.IsValid():
            img = UsdGeom.Imageable(curve_prim)
            (img.MakeVisible() if show_curve else img.MakeInvisible())

        for seg_path in state.segment_paths:
            col_prim = stage.GetPrimAtPath(f"{seg_path}/collision")
            if not col_prim or not col_prim.IsValid():
                continue
            img = UsdGeom.Imageable(col_prim)
            (img.MakeInvisible() if show_curve else img.MakeVisible())

    def _make_unique_root_path(self, stage, base_name: str) -> str:
        """Return a unique root path under /World avoiding collisions with existing cables."""
        base = re.sub(r"[^\w]", "_", base_name) or "cable"
        candidate = f"/World/{base}"
        suffix = 2
        while candidate in self._cables or stage.GetPrimAtPath(candidate).IsValid():
            candidate = f"/World/{base}_{suffix:02d}"
            suffix += 1
        return candidate

    def _get_state(self, root_path: Optional[str] = None, require: bool = True) -> Optional[CableState]:
        path = root_path or self._active_path
        state = self._cables.get(path) if path else None
        if require and not state:
            raise RuntimeError("No active cable. Create or import one first.")
        return state

    def _require_stage(self):
        stage = self._usd_context.get_stage()
        if stage is None:
            raise RuntimeError("No open USD stage. Create or open a stage before building a cable.")
        return stage

    def _infer_params_from_stage(
        self, stage, root_path: str, segment_paths: List[str], joint_paths: List[str]
    ) -> Tuple[RopeParameters, Dict[str, Dict[str, Tuple[float, float]]]]:
        radius = 0.01
        seg_len = 0.1
        mass_total = 1.0
        limits: Dict[str, Dict[str, Tuple[float, float]]] = {}

        # Radius + segment length from first segment collision.
        first_col = stage.GetPrimAtPath(f"{segment_paths[0]}/collision")
        if first_col and first_col.IsValid():
            rad_attr = first_col.GetAttribute("radius")
            if rad_attr and rad_attr.HasAuthoredValueOpinion():
                radius = float(rad_attr.Get())
            h_attr = first_col.GetAttribute("height")
            if h_attr and h_attr.HasAuthoredValueOpinion():
                seg_len = float(h_attr.Get()) + 2.0 * radius

        # Fallback segment length from spacing of first two centers.
        if len(segment_paths) >= 2:
            p0 = self._segment_world_pos(stage, segment_paths[0])
            p1 = self._segment_world_pos(stage, segment_paths[1])
            if p0 is not None and p1 is not None:
                seg_len = max(float((Gf.Vec3d(p1) - Gf.Vec3d(p0)).GetLength()), seg_len)

        # Mass from summing segment masses.
        total = 0.0
        for path in segment_paths:
            prim = stage.GetPrimAtPath(path)
            mass_api = UsdPhysics.MassAPI(prim)
            mass_attr = mass_api.GetMassAttr() if mass_api else None
            if mass_attr and mass_attr.HasAuthoredValueOpinion():
                total += float(mass_attr.Get())
        if total > 0.0:
            mass_total = total

        # Limits from joints.
        for jp in joint_paths:
            joint_limits: Dict[str, Tuple[float, float]] = {}
            joint_prim = stage.GetPrimAtPath(jp)
            for axis in ROT_AXES:
                lim_api = UsdPhysics.LimitAPI.Get(joint_prim, axis)
                low = lim_api.GetLowAttr().Get() if lim_api and lim_api.GetLowAttr() else -30.0
                high = lim_api.GetHighAttr().Get() if lim_api and lim_api.GetHighAttr() else 30.0
                joint_limits[axis] = (float(low), float(high))
            limits[jp] = joint_limits

        stiffness = DEFAULT_PARAMS.drive_stiffness
        damping = DEFAULT_PARAMS.drive_damping
        max_force = DEFAULT_PARAMS.drive_max_force
        if joint_paths:
            jp = joint_paths[0]
            joint_prim = stage.GetPrimAtPath(jp)
            drv = UsdPhysics.DriveAPI.Get(joint_prim, "rotX")
            if drv:
                st_attr = drv.GetStiffnessAttr()
                dp_attr = drv.GetDampingAttr()
                mf_attr = drv.GetMaxForceAttr()
                if st_attr and st_attr.HasAuthoredValueOpinion():
                    stiffness = float(st_attr.Get())
                if dp_attr and dp_attr.HasAuthoredValueOpinion():
                    damping = float(dp_attr.Get())
                if mf_attr and mf_attr.HasAuthoredValueOpinion():
                    max_force = float(mf_attr.Get())

        params = RopeParameters(
            length=seg_len * len(segment_paths),
            radius=radius,
            segment_count=len(segment_paths),
            mass=mass_total,
            rot_x_low=limits.get(joint_paths[0], {}).get("rotX", (-30.0, 30.0))[0] if joint_paths else -30.0,
            rot_x_high=limits.get(joint_paths[0], {}).get("rotX", (-30.0, 30.0))[1] if joint_paths else 30.0,
            rot_y_low=limits.get(joint_paths[0], {}).get("rotY", (-30.0, 30.0))[0] if joint_paths else -30.0,
            rot_y_high=limits.get(joint_paths[0], {}).get("rotY", (-30.0, 30.0))[1] if joint_paths else 30.0,
            rot_z_low=limits.get(joint_paths[0], {}).get("rotZ", (-30.0, 30.0))[0] if joint_paths else -30.0,
            rot_z_high=limits.get(joint_paths[0], {}).get("rotZ", (-30.0, 30.0))[1] if joint_paths else 30.0,
            drive_stiffness=stiffness,
            drive_damping=damping,
            drive_max_force=max_force,
            curve_width_scale=DEFAULT_PARAMS.curve_width_scale,
        )
        return params, limits
