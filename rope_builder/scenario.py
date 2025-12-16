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
from dataclasses import dataclass, field
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
    rot_limit_span: float = 60.0  # total allowed rotation, split symmetrically about 0
    rot_x_low: float = -30.0  # degrees
    rot_x_high: float = 30.0
    rot_y_low: float = -30.0
    rot_y_high: float = 30.0
    rot_z_low: float = -30.0
    rot_z_high: float = 30.0
    drive_stiffness: float = 1200.0
    drive_damping: float = 70.0
    drive_max_force: float = 200.0
    curve_extension: float = 0.02  # meters to extend spline beyond first/last collider
    curve_width_scale: float = 2.0  # multiplier for visual curve width (radius * scale)

    @property
    def segment_length(self) -> float:
        """Nominal length of a regular segment (excludes the two shortened end segments)."""
        if self.segment_count <= 0:
            return 0.0
        return self.length / float(self.segment_count + 0.5)

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
        half = max(self.rot_limit_span * 0.5, 0.0)
        return {
            "rotX": (-half, half),
            "rotY": (-half, half),
            "rotZ": (-half, half),
        }


@dataclass
class CableState:
    root_path: str
    curve_path: str
    params: RopeParameters
    segment_lengths: List[float]
    segment_paths: List[str]
    joint_paths: List[str]
    joint_limits: Dict[str, Dict[str, Tuple[float, float]]]
    joint_drive_targets: Dict[str, Dict[str, float]]
    joint_local_offsets: Dict[str, Dict[str, Any]]
    anchor_start: str
    anchor_end: str
    anchors_follow_rope: bool = True
    original_length: float = 0.0
    current_path_length: float = 0.0
    handle_paths: List[str] = field(default_factory=list)
    show_curve: bool = True
    update_subscription: Optional[Any] = None
    # Performance helpers
    dirty: bool = True
    _accum_dt: float = 0.0
    _last_endpoints: Optional[Tuple[Gf.Vec3f, Gf.Vec3f]] = None


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
        self._sync_rot_limits_from_span()
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

        segment_plan = self._segment_plan(params)
        total_len = sum(length for _, length in segment_plan)
        # Give start/end segments only a fraction of the mass of regular segments based on length ratio.
        weights = [
            (length / params.segment_length) if params.segment_length > 0.0 else 1.0 for _, length in segment_plan
        ]
        weight_sum = sum(weights) if weights else 1.0
        mass_per_weight = params.mass / weight_sum if weight_sum > 0.0 else 0.0
        cursor = -0.5 * total_len

        prev_segment_path = None
        prev_len = None
        for (name, seg_len), weight in zip(segment_plan, weights):
            center = Gf.Vec3d(cursor + 0.5 * seg_len, 0.0, 0.0)
            seg_mass = mass_per_weight * weight
            segment_path = self._create_segment(stage, root_path, params, name, seg_len, center, seg_mass)
            if prev_segment_path and prev_len is not None:
                joint_path, limits = self._create_d6_joint(
                    stage, root_path, params, len(joint_paths), prev_segment_path, segment_path, prev_len, seg_len
                )
                joint_paths.append(joint_path)
                joint_limits[joint_path] = limits
                joint_targets[joint_path] = {axis: 0.0 for axis in ROT_AXES}
            segment_paths.append(segment_path)
            prev_segment_path = segment_path
            prev_len = seg_len
            cursor += seg_len

        curve_path = f"{root_path}/curve"
        anchor_start = f"{root_path}/anchor_start"
        anchor_end = f"{root_path}/anchor_end"
        state = CableState(
            root_path=root_path,
            curve_path=curve_path,
            params=params,
             original_length=float(params.length),
             current_path_length=float(params.length),
            segment_lengths=[length for _, length in segment_plan],
            segment_paths=segment_paths,
            joint_paths=joint_paths,
            joint_limits=joint_limits,
            joint_drive_targets=joint_targets,
            joint_local_offsets={},
            anchor_start=anchor_start,
            anchor_end=anchor_end,
        )
        self._cables[root_path] = state
        state.dirty = True
        self._active_path = root_path

        self._ensure_curve_prim(stage, state)
        self._update_curve_points(state)
        self._define_anchor(stage, anchor_start)
        self._define_anchor(stage, anchor_end)
        self._update_anchors(state)
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
        # Also include start/end if named that way.
        start_path = f"{root_path}/segment_start"
        end_path = f"{root_path}/segment_end"
        if stage.GetPrimAtPath(start_path).IsValid():
            segment_paths = [start_path] + segment_paths
        if stage.GetPrimAtPath(end_path).IsValid():
            segment_paths = segment_paths + [end_path]
        if len(segment_paths) < 2:
            raise RuntimeError("Need at least 2 segments to import a cable.")

        # Joints may not be direct children in some authored cables, so search under the root.
        joint_regex = re.compile(r"joint_(\d+)$")
        joints: List[Tuple[int, str]] = []
        for prim in stage.Traverse():
            if not prim or not prim.IsValid():
                continue
            p = prim.GetPath().pathString
            if not p.startswith(root_path + "/"):
                continue
            m = joint_regex.search(prim.GetName())
            if m:
                joints.append((int(m.group(1)), p))
        joints.sort(key=lambda x: x[0])
        joint_paths = [p for _, p in joints]
        if not joint_paths:
            carb.log_warn(f"[RopeBuilder] No joints found under {root_path}. Joint UI will be empty.")

        params, joint_limits, seg_lengths = self._infer_params_from_stage(stage, root_path, segment_paths, joint_paths)
        # Seed joint drive targets from existing authored drive target positions if present.
        # If not authored (common in Isaac Sim 5.0), infer targets from the current pose so UI sliders
        # match the "Local Offset" shown in the Joint Properties panel.
        inferred_from_pose = self._infer_joint_targets_from_pose(stage, segment_paths, joint_paths)

        joint_targets: Dict[str, Dict[str, float]] = {}
        for jp in joint_paths:
            joint_targets[jp] = {}
            joint_prim = stage.GetPrimAtPath(jp)
            lims = joint_limits.get(jp, {})
            inferred_axes = inferred_from_pose.get(jp, {})

            for axis in ROT_AXES:
                target_val = None
                drv = UsdPhysics.DriveAPI.Get(joint_prim, axis)
                if drv:
                    t_attr = drv.GetTargetPositionAttr()
                    if t_attr and t_attr.HasAuthoredValueOpinion():
                        try:
                            target_val = float(t_attr.Get())
                        except Exception:
                            target_val = None

                if target_val is None:
                    target_val = float(inferred_axes.get(axis, 0.0))

                low, high = lims.get(axis, (-180.0, 180.0))
                target_val = max(min(target_val, high), low)
                joint_targets[jp][axis] = target_val

        # Cache joint local offsets (imported from USD)
        joint_local_offsets: Dict[str, Dict[str, Any]] = {}
        for jp in joint_paths:
            joint_prim = stage.GetPrimAtPath(jp)
            joint = UsdPhysics.Joint(joint_prim) if joint_prim and joint_prim.IsValid() else None
            if not joint:
                continue

            # Local positions (BODY0/BODY1 frames)
            lp0_attr = joint.GetLocalPos0Attr()
            lp1_attr = joint.GetLocalPos1Attr()
            lp0 = Gf.Vec3f(lp0_attr.Get()) if lp0_attr and lp0_attr.HasAuthoredValueOpinion() else Gf.Vec3f(0.0)
            lp1 = Gf.Vec3f(lp1_attr.Get()) if lp1_attr and lp1_attr.HasAuthoredValueOpinion() else Gf.Vec3f(0.0)

            # Local orientations (quats). Isaac Sim UI shows Euler degrees; we store both quat and Euler.
            lr0_attr = joint.GetLocalRot0Attr()
            lr1_attr = joint.GetLocalRot1Attr()
            lr0 = lr0_attr.Get() if lr0_attr and lr0_attr.HasAuthoredValueOpinion() else Gf.Quatf(1.0)
            lr1 = lr1_attr.Get() if lr1_attr and lr1_attr.HasAuthoredValueOpinion() else Gf.Quatf(1.0)

            joint_local_offsets[jp] = {
                "local_pos0": lp0,
                "local_pos1": lp1,
                "local_rot0_quat": lr0,
                "local_rot1_quat": lr1,
                "local_rot0_euler": self._quat_to_euler_xyz_deg(lr0),
                "local_rot1_euler": self._quat_to_euler_xyz_deg(lr1),
                "local_rot0_authored": bool(lr0_attr and lr0_attr.HasAuthoredValueOpinion()),
                "local_rot1_authored": bool(lr1_attr and lr1_attr.HasAuthoredValueOpinion()),
            }

        # If localRot0/1 are not authored, Isaac Sim still shows a derived "Local Offset".
        # We compute an equivalent Euler offset from current pose for UI.
        for idx, jp in enumerate(joint_paths):
            off = joint_local_offsets.get(jp)
            if not off:
                continue
            if (not off.get("local_rot0_authored", False)) and idx < len(segment_paths) - 1:
                pose0 = self._segment_frame(stage, segment_paths[idx])
                pose1 = self._segment_frame(stage, segment_paths[idx + 1])
                if pose0 and pose1:
                    qrel = pose0[1].GetInverse() * pose1[1]
                    off["local_rot0_euler"] = self._quat_to_euler_xyz_deg(qrel)
            if (not off.get("local_rot1_authored", False)):
                off.setdefault("local_rot1_euler", (0.0, 0.0, 0.0))

        # Discover any existing shape handles under this cable root so they can
        # be used for edit-mode fitting.
        handle_regex = re.compile(r"handle_(\d+)$")
        handle_pairs: List[Tuple[int, str]] = []
        for child in root_prim.GetChildren():
            name = child.GetName()
            m = handle_regex.search(name)
            if m:
                handle_pairs.append((int(m.group(1)), child.GetPath().pathString))
        handle_pairs.sort(key=lambda x: x[0])
        handle_paths = [p for _, p in handle_pairs]

        curve_path = f"{root_path}/curve"
        anchor_start = f"{root_path}/anchor_start"
        anchor_end = f"{root_path}/anchor_end"
        state = CableState(
            root_path=root_path,
            curve_path=curve_path,
            params=params,
            original_length=float(params.length),
            current_path_length=float(params.length),
            segment_lengths=seg_lengths,
            segment_paths=segment_paths,
            joint_paths=joint_paths,
            joint_limits=joint_limits,
            joint_drive_targets=joint_targets,
            joint_local_offsets=joint_local_offsets,
            anchor_start=anchor_start,
            anchor_end=anchor_end,
            handle_paths=handle_paths,
        )
        self._cables[root_path] = state
        state.dirty = True
        self._active_path = root_path

        self._ensure_curve_prim(stage, state)
        self._update_curve_points(state)
        self._define_anchor(stage, anchor_start)
        self._define_anchor(stage, anchor_end)
        self._update_anchors(state)
        self._apply_visibility_state(state)
        self._ensure_end_tip_prims(stage, state)
        carb.log_info(f"[RopeBuilder] Imported cable at {root_path}.")
        return root_path

    def _ensure_end_tip_prims(self, stage, state: CableState):
        """Ensure segment_start/tip and segment_end/tip exist for plug attachment."""
        if not state.segment_paths:
            return
        seg_lengths = state.segment_lengths or [state.params.segment_length] * len(state.segment_paths)

        # Start segment tip at local -X end.
        start_seg = state.segment_paths[0]
        start_len = seg_lengths[0] if seg_lengths else state.params.segment_length
        start_tip = Sdf.Path(start_seg).AppendPath("tip")
        if not stage.GetPrimAtPath(start_tip).IsValid():
            col_local = self._child_local_offset(stage, start_seg, f"{start_seg}/collision") or Gf.Vec3d(0.0, 0.0, 0.0)
            tip_local = col_local + Gf.Vec3d(-0.5 * float(start_len), 0.0, 0.0)
            tip = UsdGeom.Xform.Define(stage, start_tip)
            tip.AddTranslateOp().Set(Gf.Vec3f(float(tip_local[0]), float(tip_local[1]), float(tip_local[2])))
        start_attach = start_tip.AppendPath("attach")
        if not stage.GetPrimAtPath(start_attach).IsValid():
            attach = UsdGeom.Xform.Define(stage, start_attach)
            attach.AddTranslateOp().Set(Gf.Vec3f(0.0, 0.0, 0.0))

        # End segment tip at local +X end.
        end_seg = state.segment_paths[-1]
        end_len = seg_lengths[-1] if seg_lengths else state.params.segment_length
        end_tip = Sdf.Path(end_seg).AppendPath("tip")
        if not stage.GetPrimAtPath(end_tip).IsValid():
            col_local = self._child_local_offset(stage, end_seg, f"{end_seg}/collision") or Gf.Vec3d(0.0, 0.0, 0.0)
            tip_local = col_local + Gf.Vec3d(0.5 * float(end_len), 0.0, 0.0)
            tip = UsdGeom.Xform.Define(stage, end_tip)
            tip.AddTranslateOp().Set(Gf.Vec3f(float(tip_local[0]), float(tip_local[1]), float(tip_local[2])))
        end_attach = end_tip.AppendPath("attach")
        if not stage.GetPrimAtPath(end_attach).IsValid():
            attach = UsdGeom.Xform.Define(stage, end_attach)
            attach.AddTranslateOp().Set(Gf.Vec3f(0.0, 0.0, 0.0))

    def delete_rope(self, root_path: Optional[str] = None):
        """Remove a cable from the stage and controller."""
        target_path = root_path or self._active_path or (self.list_cable_paths()[0] if self._cables else None)
        state = self._get_state(target_path, require=False)
        if not state:
            return

        self.stop_curve_updates(state.root_path)

        stage = self._usd_context.get_stage()
        if stage:
            # Preserve anchors by moving them under /World before deleting the cable root.
            for anchor in (state.anchor_start, state.anchor_end):
                self._move_anchor_to_world(stage, anchor)
            prim = stage.GetPrimAtPath(state.root_path)
            if prim and prim.IsValid():
                stage.RemovePrim(state.root_path)
                carb.log_info(f"[RopeBuilder] Deleted cable prim hierarchy at {state.root_path}.")

        self._cables.pop(state.root_path, None)
        if self._active_path == state.root_path:
            self._active_path = self.list_cable_paths()[0] if self._cables else None

    def forget_all_cables(self):
        """Drop all cached cable state without touching stage prims (used on stage reload/exit)."""
        for root_path in list(self._cables.keys()):
            self.stop_curve_updates(root_path)
        self._cables.clear()
        self._active_path = None

    def validate_parameters(self) -> bool:
        return self._validate_params(self._params)

    def any_curve_subscription_active(self) -> bool:
        """True if at least one cable has an update subscription."""
        for state in self._cables.values():
            if state.update_subscription:
                return True
        return False

    def start_curve_updates_all(self):
        """Subscribe spline updates for all cables."""
        for rp in self.list_cable_paths():
            try:
                self.start_curve_updates(rp)
            except Exception as exc:
                carb.log_warn(f"[RopeBuilder] Failed to start curve updates for {rp}: {exc}")

    def stop_curve_updates_all(self):
        """Unsubscribe spline updates for all cables."""
        for rp in self.list_cable_paths():
            try:
                self.stop_curve_updates(rp)
            except Exception as exc:
                carb.log_warn(f"[RopeBuilder] Failed to stop curve updates for {rp}: {exc}")

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
        state.dirty = True
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
    
    def showing_curve_state(self) -> bool:
        """Return current show_curve state for the 'global' UI toggle."""
        paths = self.list_cable_paths()
        if self._active_path and self._active_path in self._cables:
            return self._cables[self._active_path].show_curve
        if paths:
            return self._cables[paths[0]].show_curve
        return True  # default if none exist

    def toggle_visibility_all(self) -> bool:
        """Toggle spline/collision visibility for ALL known cables."""
        paths = self.list_cable_paths()
        if not paths:
            # Nothing to toggle; keep default spline-visible state.
            return True

        current = self.showing_curve_state()
        new_show_curve = not current

        for rp in paths:
            state = self._cables.get(rp)
            if not state:
                continue
            state.show_curve = new_show_curve
            state.dirty = True
            self._apply_visibility_state(state)

        return new_show_curve


    def toggle_visibility(self) -> bool:
        """Toggle between showing spline or collision capsules on the active cable."""
        state = self._get_state()
        state.show_curve = not state.show_curve
        state.dirty = True        
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

    def create_shape_handle(self, root_path: Optional[str] = None) -> str:
        """Create a movable handle prim for shaping the cable between fixed anchors."""
        state = self._get_state(root_path, require=True)
        stage = self._require_stage()

        # Default position: midpoint between anchor transforms, or cable root if anchors invalid.
        pos = Gf.Vec3d(0.0, 0.0, 0.0)
        start_pose = self._segment_frame(stage, state.anchor_start)
        end_pose = self._segment_frame(stage, state.anchor_end)
        if start_pose and end_pose:
            pos = (start_pose[0] + end_pose[0]) * 0.5
        else:
            root_prim = stage.GetPrimAtPath(state.root_path)
            if root_prim and root_prim.IsValid():
                xf = UsdGeom.Xformable(root_prim)
                m = xf.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                pos = Gf.Vec3d(m.ExtractTranslation())

        # Choose a unique handle path under the cable root.
        base_name = "handle"
        index = len(getattr(state, "handle_paths", []))
        candidate = f"{state.root_path}/{base_name}_{index:02d}"
        suffix = index + 1
        while stage.GetPrimAtPath(candidate).IsValid():
            candidate = f"{state.root_path}/{base_name}_{suffix:02d}"
            suffix += 1
        handle_path = candidate

        xform = UsdGeom.Xform.Define(stage, Sdf.Path(handle_path))
        xf = UsdGeom.Xformable(xform.GetPrim())
        xf.ClearXformOpOrder()
        xf.AddTranslateOp().Set(Gf.Vec3f(pos))

        # Optional small visual so the handle is easy to select.
        try:
            sphere_path = Sdf.Path(handle_path).AppendPath("visual")
            sphere = UsdGeom.Sphere.Define(stage, sphere_path)
            sphere.CreateRadiusAttr(0.01)
        except Exception:
            pass

        state.handle_paths.append(handle_path)
        carb.log_info(f"[RopeBuilder] Created shape handle at {handle_path}.")
        return handle_path

    def get_joint_limit_violations(self, root_path: Optional[str] = None) -> Tuple[int, float]:
        """Return (num_axes_violating_limits, max_violation_deg) for the given or active cable."""
        state = self._get_state(root_path, require=False)
        if not state:
            return 0, 0.0

        count = 0
        max_over = 0.0
        for joint_path in state.joint_paths:
            limits = state.joint_limits.get(joint_path, {})
            targets = state.joint_drive_targets.get(joint_path, {})
            for axis in ROT_AXES:
                val = float(targets.get(axis, 0.0))
                low, high = limits.get(axis, (-180.0, 180.0))
                if val < low:
                    over = float(low - val)
                elif val > high:
                    over = float(val - high)
                else:
                    over = 0.0
                if over > 0.0:
                    count += 1
                    if over > max_over:
                        max_over = over
        return count, max_over

    def get_length_info(self, root_path: Optional[str] = None) -> Tuple[float, float]:
        """Return (original_length, current_path_length) for the given or active cable."""
        state = self._get_state(root_path, require=False)
        if not state:
            return 0.0, 0.0
        return float(getattr(state, "original_length", 0.0)), float(getattr(state, "current_path_length", 0.0))

    def set_shape_handles_visible_all(self, visible: bool):
        """Show or hide all known shape handles across all cables."""
        stage = self._usd_context.get_stage()
        if not stage:
            return

        for state in self._cables.values():
            for hpath in getattr(state, "handle_paths", []) or []:
                prim = stage.GetPrimAtPath(hpath)
                if not prim or not prim.IsValid():
                    continue
                img = UsdGeom.Imageable(prim)
                img.MakeVisible() if visible else img.MakeInvisible()

    def fit_rope_to_anchors(self, root_path: Optional[str] = None) -> Optional[Tuple[float, float]]:
        """Repose the cable along a geometric curve between anchors in edit mode.

        This preserves per-segment lengths, ignores joint limits, and may introduce
        error if the anchor path length does not match the cable length. The caller
        can use the returned (rope_length, path_length) to display a warning.
        """
        state = self._get_state(root_path, require=False)
        stage = self._usd_context.get_stage()
        if not state or not stage or not state.segment_paths:
            return None

        # Switch to anchor-driven mode: anchors become user handles and stop
        # being overwritten from cable tips.
        try:
            state.anchors_follow_rope = False
        except Exception:
            pass

        # Anchor prims are used as authoring handles: read their current world-space
        # frames and build a smooth curve between them, optionally passing through
        # user-created shape handles.
        start_pose = self._segment_frame(stage, state.anchor_start)
        end_pose = self._segment_frame(stage, state.anchor_end)
        if not start_pose or not end_pose:
            carb.log_warn("[RopeBuilder] Cannot fit cable: anchors not found or invalid.")
            return None

        p0, r0 = start_pose
        p1, r1 = end_pose

        seg_lengths = state.segment_lengths or [state.params.segment_length] * len(state.segment_paths)
        rope_len = float(sum(seg_lengths)) if seg_lengths else float(state.params.length)
        if rope_len <= 1e-6:
            carb.log_warn("[RopeBuilder] Cannot fit cable: total cable length is zero.")
            return None

        # Tangent directions derived from anchor orientations (their +X axis).
        dir0 = Gf.Rotation(r0).TransformDir(Gf.Vec3d(1.0, 0.0, 0.0))
        dir1 = Gf.Rotation(r1).TransformDir(Gf.Vec3d(1.0, 0.0, 0.0))
        if dir0.GetLength() < 1e-6:
            dir0 = Gf.Vec3d(1.0, 0.0, 0.0)
        if dir1.GetLength() < 1e-6:
            dir1 = Gf.Vec3d(1.0, 0.0, 0.0)

        # Align end segments exactly to anchor transforms. The interior curve is fit
        # between the inner ends of the end segments (i.e. the first/last joints).
        start_len = float(seg_lengths[0]) if seg_lengths else 0.0
        end_len = float(seg_lengths[-1]) if seg_lengths else 0.0

        # If the user moved tip/attach to a plug mating point, the anchor targets that
        # point (not necessarily the segment tip). Compute the distance from the anchor
        # (attach) to the inner joint using local offsets, so the rest of the cable
        # starts/ends at the correct locations.
        start_seg_path = state.segment_paths[0]
        end_seg_path = state.segment_paths[-1]

        start_col_local = self._child_local_offset(stage, start_seg_path, f"{start_seg_path}/collision") or Gf.Vec3d(
            0.0, 0.0, 0.0
        )
        end_col_local = self._child_local_offset(stage, end_seg_path, f"{end_seg_path}/collision") or Gf.Vec3d(
            0.0, 0.0, 0.0
        )

        start_attach_local = self._child_local_offset(stage, start_seg_path, f"{start_seg_path}/tip/attach")
        if start_attach_local is None:
            start_attach_local = self._child_local_offset(stage, start_seg_path, f"{start_seg_path}/tip")
        if start_attach_local is None:
            start_attach_local = start_col_local + Gf.Vec3d(-0.5 * start_len, 0.0, 0.0)

        end_attach_local = self._child_local_offset(stage, end_seg_path, f"{end_seg_path}/tip/attach")
        if end_attach_local is None:
            end_attach_local = self._child_local_offset(stage, end_seg_path, f"{end_seg_path}/tip")
        if end_attach_local is None:
            end_attach_local = end_col_local + Gf.Vec3d(0.5 * end_len, 0.0, 0.0)

        start_inner_local = start_col_local + Gf.Vec3d(0.5 * start_len, 0.0, 0.0)
        end_inner_local = end_col_local + Gf.Vec3d(-0.5 * end_len, 0.0, 0.0)
        start_to_inner = float(start_inner_local[0] - start_attach_local[0])
        end_to_inner = float(end_attach_local[0] - end_inner_local[0])

        inner_p0 = p0 + dir0 * start_to_inner
        inner_p1 = p1 - dir1 * end_to_inner

        delta = inner_p1 - inner_p0
        straight_dist = delta.GetLength()
        # Choose a tangent magnitude that gives a gentle arc; fall back to rope_len
        # for nearly coincident anchors.
        tangent_scale = straight_dist * 0.5
        if tangent_scale < 1e-4:
            tangent_scale = max(rope_len * 0.25, 1e-3)
        m0 = dir0 * tangent_scale
        m1 = dir1 * tangent_scale

        # Collect control points for the interior curve: inner-start tip, any shape
        # handles, inner-end tip.
        ctrl_pts: List[Gf.Vec3d] = [inner_p0]
        for hpath in getattr(state, "handle_paths", []) or []:
            prim = stage.GetPrimAtPath(hpath)
            if not prim or not prim.IsValid():
                continue
            xf = UsdGeom.Xformable(prim)
            m = xf.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            ctrl_pts.append(Gf.Vec3d(m.ExtractTranslation()))
        ctrl_pts.append(inner_p1)

        # For Catmull-Rom, create extended endpoint samples to respect anchor tangents.
        n_ctrl = len(ctrl_pts)
        if n_ctrl < 2:
            ctrl_pts = [inner_p0, inner_p1]
            n_ctrl = 2

        ext_pts: List[Gf.Vec3d] = [Gf.Vec3d(0.0)] * (n_ctrl + 2)
        for i, p in enumerate(ctrl_pts):
            ext_pts[i + 1] = p
        ext_pts[0] = ctrl_pts[0] - dir0 * tangent_scale
        ext_pts[-1] = ctrl_pts[-1] + dir1 * tangent_scale

        num_segs = max(n_ctrl - 1, 1)

        def catmull_point_and_tangent(u: float) -> Tuple[Gf.Vec3d, Gf.Vec3d]:
            """Evaluate Catmull-Rom spline point and tangent for u in [0, 1]."""
            if num_segs <= 0:
                return Gf.Vec3d(ctrl_pts[0]), Gf.Vec3d(dir0)

            u = max(0.0, min(1.0, float(u)))
            s = u * float(num_segs)
            seg = int(s)
            if seg >= num_segs:
                seg = num_segs - 1
                t = 1.0
            else:
                t = s - float(seg)

            i = seg  # local segment index between ctrl_pts[i] and ctrl_pts[i+1]
            P0 = ext_pts[i]
            P1 = ext_pts[i + 1]
            P2 = ext_pts[i + 2]
            P3 = ext_pts[i + 3]

            t2 = t * t
            t3 = t2 * t

            # Standard Catmull-Rom position.
            pos = 0.5 * (
                (2.0 * P1)
                + (-P0 + P2) * t
                + (2.0 * P0 - 5.0 * P1 + 4.0 * P2 - P3) * t2
                + (-P0 + 3.0 * P1 - 3.0 * P2 + P3) * t3
            )

            # Derivative for tangent.
            dpos = 0.5 * (
                (-P0 + P2)
                + (2.0 * (2.0 * P0 - 5.0 * P1 + 4.0 * P2 - P3)) * t
                + (3.0 * (-P0 + 3.0 * P1 - 3.0 * P2 + P3)) * t2
            )

            if dpos.GetLength() < 1e-6:
                dpos = Gf.Vec3d(dir0)
            else:
                dpos.Normalize()
            return pos, dpos

        # Sample the curve to approximate arc length for parameterization.
        num_samples = max(32, len(state.segment_paths) * 4)
        ts: List[float] = []
        pts: List[Gf.Vec3d] = []
        cumulative: List[float] = []

        last_pos = None
        length_accum = 0.0
        for i in range(num_samples):
            t = float(i) / float(max(num_samples - 1, 1))
            pos, _ = catmull_point_and_tangent(t)
            ts.append(t)
            pts.append(pos)
            if last_pos is not None:
                length_accum += float((pos - last_pos).GetLength())
            cumulative.append(length_accum)
            last_pos = pos

        curve_len = float(length_accum)
        use_straight = curve_len <= 1e-6
        if use_straight:
            # Degenerate curve; fall back to straight line between anchors.
            curve_len = float(straight_dist)

        def sample_pos_and_dir(s: float) -> Tuple[Gf.Vec3d, Gf.Vec3d]:
            """Return a point and approximate tangent direction at normalized arc-length s."""
            s_clamped = max(0.0, min(1.0, float(s)))
            if use_straight or curve_len <= 1e-6:
                pos = inner_p0 + delta * s_clamped
                d = Gf.Vec3d(delta)
                if d.GetLength() < 1e-6:
                    d = dir0
                else:
                    d.Normalize()
                return pos, d

            target_len = s_clamped * curve_len
            # Find the first sample with cumulative length >= target_len.
            idx = 0
            while idx < len(cumulative) and cumulative[idx] < target_len:
                idx += 1

            if idx <= 0:
                t_val = ts[0]
            elif idx >= len(cumulative):
                t_val = ts[-1]
            else:
                l0 = cumulative[idx - 1]
                l1 = cumulative[idx]
                if l1 <= l0:
                    alpha = 0.0
                else:
                    alpha = (target_len - l0) / (l1 - l0)
                t_val = ts[idx - 1] * (1.0 - alpha) + ts[idx] * alpha

            pos, tan = catmull_point_and_tangent(t_val)
            return pos, tan

        # Lay out segments:
        # - End segments match anchors exactly (for plug alignment).
        # - Interior segments follow the sampled interior curve.
        mid_rope_len = float(rope_len - start_len - end_len)
        cursor_mid = 0.0
        last_index = len(state.segment_paths) - 1

        # Robustly align end segments by their authored tip prims (if present). This avoids
        # half-segment offsets when segment xform origins are not centered (common in imports).
        start_seg_path = state.segment_paths[0]
        end_seg_path = state.segment_paths[-1]
        start_attach_local = self._child_local_offset(stage, start_seg_path, f"{start_seg_path}/tip/attach")
        end_attach_local = self._child_local_offset(stage, end_seg_path, f"{end_seg_path}/tip/attach")
        start_tip_local = self._child_local_offset(stage, start_seg_path, f"{start_seg_path}/tip")
        end_tip_local = self._child_local_offset(stage, end_seg_path, f"{end_seg_path}/tip")
        for idx, seg_path in enumerate(state.segment_paths):
            seg_len = seg_lengths[idx] if idx < len(seg_lengths) else state.params.segment_length

            desired_world_q: Optional[Gf.Quatd] = None
            seg_origin_world: Optional[Gf.Vec3d] = None

            # End segments: force exact alignment with anchor transforms so plugs
            # parented under segment_start/tip and segment_end/tip match orientation.
            if idx == 0:
                tangent_world = dir0
                desired_world_q = r0
                tip_local = (
                    start_attach_local
                    if start_attach_local is not None
                    else (start_tip_local if start_tip_local is not None else Gf.Vec3d(-0.5 * float(seg_len), 0.0, 0.0))
                )
                seg_origin_world = p0 - Gf.Rotation(desired_world_q).TransformDir(tip_local)
            elif idx == last_index:
                tangent_world = dir1
                desired_world_q = r1
                tip_local = (
                    end_attach_local
                    if end_attach_local is not None
                    else (end_tip_local if end_tip_local is not None else Gf.Vec3d(0.5 * float(seg_len), 0.0, 0.0))
                )
                seg_origin_world = p1 - Gf.Rotation(desired_world_q).TransformDir(tip_local)
            else:
                # Interior segments follow the smooth sampled curve.
                if mid_rope_len <= 1e-6:
                    mid_s = 0.5
                else:
                    mid_s = (cursor_mid + 0.5 * float(seg_len)) / mid_rope_len
                center_world, tangent_world = sample_pos_and_dir(mid_s)

                # Interior segments: +X axis points along tangent_world.
                world_rot = Gf.Rotation(Gf.Vec3d(1.0, 0.0, 0.0), tangent_world).GetQuat()
                desired_world_q = Gf.Quatd(world_rot.GetReal(), world_rot.GetImaginary())

                # Place the segment so its collision prim (if any) sits on the fitted curve.
                col_local = self._child_local_offset(stage, seg_path, f"{seg_path}/collision") or Gf.Vec3d(0.0, 0.0, 0.0)
                seg_origin_world = center_world - Gf.Rotation(desired_world_q).TransformDir(col_local)

            prim = stage.GetPrimAtPath(seg_path)
            if prim and prim.IsValid():
                xf = UsdGeom.Xformable(prim)

                # Convert world-space center/orientation into the local space of the parent
                # so moving the cable root does not introduce an offset.
                parent = prim.GetParent()
                parent_world = Gf.Matrix4d(1.0)
                if parent and parent.IsValid():
                    parent_xf = UsdGeom.Xformable(parent)
                    parent_world = parent_xf.ComputeLocalToWorldTransform(Usd.TimeCode.Default())

                inv_parent = parent_world.GetInverse()
                if seg_origin_world is None:
                    seg_origin_world = Gf.Vec3d(0.0, 0.0, 0.0)
                local_pos = inv_parent.Transform(seg_origin_world)

                parent_rot = parent_world.ExtractRotation().GetQuat()
                local_q = parent_rot.GetInverse() * desired_world_q

                xf.ClearXformOpOrder()
                xf.AddTranslateOp().Set(Gf.Vec3f(local_pos))
                qf = Gf.Quatf(float(local_q.GetReal()), Gf.Vec3f(local_q.GetImaginary()))
                xf.AddOrientOp().Set(qf)
                xf.AddScaleOp().Set(Gf.Vec3f(1.0, 1.0, 1.0))

            if idx != 0 and idx != last_index:
                cursor_mid += float(seg_len)

        # Sanity check: fitted end segment tips should land on the anchors.
        try:
            tip0 = self._prim_world_pos(stage, f"{start_seg_path}/tip/attach") or self._prim_world_pos(
                stage, f"{start_seg_path}/tip"
            )
            tip1 = self._prim_world_pos(stage, f"{end_seg_path}/tip/attach") or self._prim_world_pos(
                stage, f"{end_seg_path}/tip"
            )
            eps = 1e-4
            if tip0 is not None and float((tip0 - p0).GetLength()) > eps:
                carb.log_warn(
                    f"[RopeBuilder] Start attach mismatch after fit: {float((tip0 - p0).GetLength()):.6f} m"
                )
            if tip1 is not None and float((tip1 - p1).GetLength()) > eps:
                carb.log_warn(f"[RopeBuilder] End attach mismatch after fit: {float((tip1 - p1).GetLength()):.6f} m")
        except Exception:
            pass

        # Update spline points from new segment poses, but keep anchors as user-authored
        # handles (do not overwrite them from segment endpoints here).
        self._update_curve_points(state)

        # Refresh endpoint cache for curve update throttling.
        if stage and state.segment_paths:
            p0d = self._prim_world_pos(stage, f"{state.segment_paths[0]}/collision") or self._prim_world_pos(
                stage, state.segment_paths[0]
            )
            p1d = self._prim_world_pos(stage, f"{state.segment_paths[-1]}/collision") or self._prim_world_pos(
                stage, state.segment_paths[-1]
            )
            if p0d is not None and p1d is not None:
                state._last_endpoints = (
                    Gf.Vec3f(float(p0d[0]), float(p0d[1]), float(p0d[2])),
                    Gf.Vec3f(float(p1d[0]), float(p1d[1]), float(p1d[2])),
                )
        state.dirty = False

        # Update cached joint drive targets from the new pose so the edit UI stays in sync.
        try:
            inferred = self._infer_joint_targets_from_pose(stage, state.segment_paths, state.joint_paths)
            for jp, axes in inferred.items():
                state.joint_drive_targets[jp] = dict(axes)
        except Exception as exc:
            carb.log_warn(f"[RopeBuilder] Failed to infer joint targets after fitting to anchors: {exc}")

        # Store length info for UI (original is the designed cable length; current is the fitted path length).
        path_len = float(start_len + curve_len + end_len)
        state.original_length = float(rope_len)
        state.current_path_length = path_len

        return rope_len, path_len

    # ----------------------------------------------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------------------------------------------

    def discover_cables(self, prefix: str = "/World") -> List[str]:
        """Discover existing cable roots in the stage and import them."""
        stage = self._require_stage()

        seg_regex = re.compile(r"segment_(\d+)$")
        found: List[str] = []

        for prim in stage.Traverse():
            if not prim or not prim.IsValid():
                continue

            path = prim.GetPath().pathString
            if prefix and not path.startswith(prefix):
                continue

            # detect a cable root by having segment children
            has_segment = False
            for child in prim.GetChildren():
                name = child.GetName()
                if name in ("segment_start", "segment_end") or seg_regex.search(name):
                    has_segment = True
                    break

            if not has_segment:
                continue

            if path not in self._cables:
                try:
                    self.import_cable(path)
                    found.append(path)
                except Exception:
                    # ignore prims that look similar but aren't valid cables
                    pass

        return found

    def _ensure_parameter_defaults(self):
        """Fill in any missing attributes when hot-reloading older state."""
        had_span = hasattr(self._params, "rot_limit_span")
        for field in DEFAULT_PARAMS.__dataclass_fields__.keys():
            if not hasattr(self._params, field):
                setattr(self._params, field, getattr(DEFAULT_PARAMS, field))

        # Older cached params won't have rot_limit_span. Derive it from X limits if missing.
        if not had_span:
            try:
                span = float(getattr(self._params, "rot_x_high", 30.0) - getattr(self._params, "rot_x_low", -30.0))
            except Exception:
                span = DEFAULT_PARAMS.rot_limit_span
            self._params.rot_limit_span = max(span, 0.0)

        self._sync_rot_limits_from_span()

    def _sync_rot_limits_from_span(self):
        """Keep per-axis limits symmetric based on the configured span."""
        half = max(float(getattr(self._params, "rot_limit_span", DEFAULT_PARAMS.rot_limit_span)) * 0.5, 0.0)
        lows = (-half, -half, -half)
        highs = (half, half, half)
        self._params.rot_x_low, self._params.rot_y_low, self._params.rot_z_low = lows
        self._params.rot_x_high, self._params.rot_y_high, self._params.rot_z_high = highs

    @staticmethod
    def _validate_params(params: RopeParameters) -> bool:
        """Basic sanity checks to catch common input mistakes."""
        segment_length = params.segment_length
        short_length = segment_length * 0.25
        return all(
            [
                params.length > 0.0,
                params.radius > 0.0,
                params.segment_count > 1,
                params.mass > 0.0,
                segment_length > params.radius * 2.0,
                short_length > params.radius * 2.0,
                params.rot_x_low < params.rot_x_high,
                params.rot_y_low < params.rot_y_high,
                params.rot_z_low < params.rot_z_high,
                params.drive_stiffness >= 0.0,
                params.drive_damping >= 0.0,
                params.drive_max_force >= 0.0,
                params.curve_extension >= 0.0,
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

    def _create_segment(
        self,
        stage,
        root_path: str,
        params: RopeParameters,
        name: str,
        seg_len: float,
        center: Gf.Vec3d,
        seg_mass: float,
    ) -> str:
        """Author the prims for an individual cable segment (rigid body + collider only)."""
        segment_path = Sdf.Path(f"{root_path}/{name}")
        xform = UsdGeom.Xform.Define(stage, segment_path)

        xform.AddTranslateOp().Set(center)

        rigid_prim = xform.GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(rigid_prim)
        mass_api = UsdPhysics.MassAPI.Apply(rigid_prim)
        mass_api.CreateMassAttr(seg_mass)

        physx_body = PhysxSchema.PhysxRigidBodyAPI.Apply(rigid_prim)
        if physx_body:
            physx_body.CreateEnableCCDAttr(True)

        collision_path = segment_path.AppendPath("collision")
        collision = UsdGeom.Capsule.Define(stage, collision_path)
        collision.CreateRadiusAttr(params.radius)
        collision.CreateHeightAttr(max(seg_len - 2.0 * params.radius, 1e-4))
        collision.CreateAxisAttr(UsdGeom.Tokens.x)
        UsdPhysics.CollisionAPI.Apply(collision.GetPrim())
        UsdGeom.Imageable(collision.GetPrim()).MakeInvisible()

        # End-of-cable attachment points for plug meshes.
        # Users can parent plug meshes (and collider-only plug geometry) under these tips
        # so plugs move with the end segment rigid bodies without adding extra constraints.
        if name in ("segment_start", "segment_end"):
            tip_path = segment_path.AppendPath("tip")
            if not stage.GetPrimAtPath(tip_path).IsValid():
                tip = UsdGeom.Xform.Define(stage, tip_path)
                local_x = (-0.5 * seg_len) if name == "segment_start" else (0.5 * seg_len)
                tip.AddTranslateOp().Set(Gf.Vec3f(local_x, 0.0, 0.0))
            # Optional user-authored attachment point under the tip. If present, the fitter
            # can align this point (e.g. plug mating face) to the anchors instead of the tip origin.
            attach_path = tip_path.AppendPath("attach")
            if not stage.GetPrimAtPath(attach_path).IsValid():
                attach = UsdGeom.Xform.Define(stage, attach_path)
                attach.AddTranslateOp().Set(Gf.Vec3f(0.0, 0.0, 0.0))

        return str(segment_path)

    def _create_d6_joint(
        self,
        stage,
        root_path: str,
        params: RopeParameters,
        joint_index: int,
        body0_path: str,
        body1_path: str,
        len0: float,
        len1: float,
    ) -> Tuple[str, Dict[str, Tuple[float, float]]]:
        """Create a D6 joint connecting two neighboring segments."""
        joint_path = Sdf.Path(f"{root_path}/joint_{joint_index:02d}")
        joint_prim = UsdPhysics.Joint.Define(stage, joint_path).GetPrim()
        joint = UsdPhysics.Joint(joint_prim)
        joint.CreateBody0Rel().SetTargets([body0_path])
        joint.CreateBody1Rel().SetTargets([body1_path])

        joint.CreateLocalPos0Attr().Set(Gf.Vec3f(len0 * 0.5, 0.0, 0.0))
        joint.CreateLocalPos1Attr().Set(Gf.Vec3f(-len1 * 0.5, 0.0, 0.0))

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

    def _ensure_curve_prim(self, stage, state: CableState):
        curve_prim = UsdGeom.BasisCurves.Get(stage, Sdf.Path(state.curve_path))
        if not curve_prim:
            curve_prim = UsdGeom.BasisCurves.Define(stage, Sdf.Path(state.curve_path)).GetPrim()
        curves = UsdGeom.BasisCurves(curve_prim)
        type_attr = curves.GetTypeAttr() or curves.CreateTypeAttr()
        basis_attr = curves.GetBasisAttr() or curves.CreateBasisAttr()
        wrap_attr = curves.GetWrapAttr() or curves.CreateWrapAttr()

        type_attr.Set(UsdGeom.Tokens.cubic)
        basis_attr.Set(UsdGeom.Tokens.bspline)
        wrap_attr.Set(UsdGeom.Tokens.pinned)
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

        pts_world: List[Gf.Vec3d] = []

        seg_lengths = state.segment_lengths or [state.params.segment_length] * len(state.segment_paths)
        first_pose = self._segment_frame(stage, state.segment_paths[0]) if state.segment_paths else None
        last_pose = self._segment_frame(stage, state.segment_paths[-1]) if state.segment_paths else None
        half_len_first = seg_lengths[0] * 0.5 if seg_lengths else state.params.segment_length * 0.5
        half_len_last = seg_lengths[-1] * 0.5 if seg_lengths else state.params.segment_length * 0.5
        extension = max(state.params.curve_extension, 0.0)

        if first_pose:
            pos, rot = first_pose
            dir_x = Gf.Rotation(rot).TransformDir(Gf.Vec3d(1.0, 0.0, 0.0))
            tip = self._prim_world_pos(stage, f"{state.segment_paths[0]}/tip")
            if tip is None:
                tip = pos - dir_x * half_len_first
            pts_world.append(tip - dir_x * extension)

        for path in state.segment_paths:
            # Prefer the collision prim position, which remains centered even if the segment
            # xform origin is authored at an end (common in imported assets).
            wp = self._prim_world_pos(stage, f"{path}/collision")
            if wp is None:
                wp = self._prim_world_pos(stage, path)
            if wp is None:
                carb.log_warn(f"[RopeBuilder] Missing segment for curve update: {path}")
                continue
            pts_world.append(Gf.Vec3d(wp))

        if last_pose:
            pos, rot = last_pose
            dir_x = Gf.Rotation(rot).TransformDir(Gf.Vec3d(1.0, 0.0, 0.0))
            tip = self._prim_world_pos(stage, f"{state.segment_paths[-1]}/tip")
            if tip is None:
                tip = pos + dir_x * half_len_last
            pts_world.append(tip + dir_x * extension)

        curves = UsdGeom.BasisCurves(curve_prim)

        counts_attr = curves.GetCurveVertexCountsAttr()
        if not counts_attr:
            counts_attr = curves.CreateCurveVertexCountsAttr()

        points_attr = curves.GetPointsAttr()
        if not points_attr:
            points_attr = curves.CreatePointsAttr()

        if len(pts_world) < 2:
            counts_attr.Set(Vt.IntArray([0]))
            points_attr.Set(Vt.Vec3fArray())
            return

        local_pts = self._world_to_local_points(curve_prim, pts_world)
        counts_attr.Set(Vt.IntArray([len(local_pts)]))
        points_attr.Set(Vt.Vec3fArray(local_pts))

    def _on_curve_update(self, root_path: str, _dt):
        state = self._cables.get(root_path)
        if not state:
            return

        # Throttle updates to reduce idle cost.
        # In Isaac Sim 5.0 the update stream may pass an event object, not a float dt.
        try:
            dt = float(_dt) if _dt is not None else 0.0
        except Exception:
            # Fallback to an estimated frame dt so throttling still progresses.
            dt = 1.0 / 60.0

        state._accum_dt += dt
        if state._accum_dt < (1.0 / 30.0):  # ~30 Hz
            return
        state._accum_dt = 0.0

        stage = self._usd_context.get_stage()

        # Cheap movement detection if not marked dirty.
        if not state.dirty:
            if stage and state.segment_paths:
                p0d = self._prim_world_pos(stage, f"{state.segment_paths[0]}/collision") or self._prim_world_pos(
                    stage, state.segment_paths[0]
                )
                p1d = self._prim_world_pos(stage, f"{state.segment_paths[-1]}/collision") or self._prim_world_pos(
                    stage, state.segment_paths[-1]
                )
                if p0d is not None and p1d is not None:
                    p0 = Gf.Vec3f(float(p0d[0]), float(p0d[1]), float(p0d[2]))
                    p1 = Gf.Vec3f(float(p1d[0]), float(p1d[1]), float(p1d[2]))
                    last = state._last_endpoints
                    eps = 1e-5
                    if (
                        last is None
                        or (p0 - last[0]).GetLength() > eps
                        or (p1 - last[1]).GetLength() > eps
                    ):
                        state.dirty = True
                        state._last_endpoints = (p0, p1)

            if not state.dirty:
                return

        # Full update when dirty.
        self._update_curve_points(state)
        self._update_anchors(state)

        # Refresh endpoint cache after update.
        if stage and state.segment_paths:
            p0d = self._prim_world_pos(stage, f"{state.segment_paths[0]}/collision") or self._prim_world_pos(
                stage, state.segment_paths[0]
            )
            p1d = self._prim_world_pos(stage, f"{state.segment_paths[-1]}/collision") or self._prim_world_pos(
                stage, state.segment_paths[-1]
            )
            if p0d is not None and p1d is not None:
                state._last_endpoints = (
                    Gf.Vec3f(float(p0d[0]), float(p0d[1]), float(p0d[2])),
                    Gf.Vec3f(float(p1d[0]), float(p1d[1]), float(p1d[2])),
                )

        state.dirty = False

    def _segment_world_pos(self, stage, path: str) -> Optional[Gf.Vec3f]:
        prim = stage.GetPrimAtPath(path)
        if not prim or not prim.IsValid():
            return None
        xf = UsdGeom.Xformable(prim)
        m = xf.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        pos = m.ExtractTranslation()
        return Gf.Vec3f(pos[0], pos[1], pos[2])

    def _prim_world_pos(self, stage, path: str) -> Optional[Gf.Vec3d]:
        prim = stage.GetPrimAtPath(path)
        if not prim or not prim.IsValid():
            return None
        xf = UsdGeom.Xformable(prim)
        m = xf.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        return Gf.Vec3d(m.ExtractTranslation())

    def _child_local_offset(self, stage, parent_path: str, child_path: str) -> Optional[Gf.Vec3d]:
        """Return the child prim origin in the local space of parent_path."""
        parent_prim = stage.GetPrimAtPath(parent_path)
        child_prim = stage.GetPrimAtPath(child_path)
        if not parent_prim or not parent_prim.IsValid() or not child_prim or not child_prim.IsValid():
            return None

        parent_xf = UsdGeom.Xformable(parent_prim)
        child_xf = UsdGeom.Xformable(child_prim)
        parent_world = parent_xf.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        child_world = child_xf.ComputeLocalToWorldTransform(Usd.TimeCode.Default())

        inv_parent = parent_world.GetInverse()
        return Gf.Vec3d(inv_parent.Transform(Gf.Vec3d(child_world.ExtractTranslation())))

    def _segment_frame(self, stage, path: str) -> Optional[Tuple[Gf.Vec3d, Gf.Quatd]]:
        prim = stage.GetPrimAtPath(path)
        if not prim or not prim.IsValid():
            return None
        xf = UsdGeom.Xformable(prim)
        m = xf.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        pos = Gf.Vec3d(m.ExtractTranslation())
        rot = m.ExtractRotation().GetQuat()
        return pos, rot

    def _world_to_local_points(self, prim, world_pts: List[Gf.Vec3d]) -> List[Gf.Vec3f]:
        """Transform world-space points into the local space of prim."""
        xf = UsdGeom.Xformable(prim)
        m_world = xf.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        inv = m_world.GetInverse()
        return [Gf.Vec3f(inv.Transform(p)) for p in world_pts]

    def _apply_edit_pose_from_targets(self, state: CableState):
        """Reposition segments in edit mode based on current joint drive targets."""
        stage = self._usd_context.get_stage()
        if not stage or not state.segment_paths:
            return

        seg_lengths = state.segment_lengths or [state.params.segment_length] * len(state.segment_paths)
        total_len = sum(seg_lengths)
        start_pt = Gf.Vec3d(-0.5 * total_len, 0.0, 0.0)
        orientation = Gf.Quatd(1.0, 0.0, 0.0, 0.0)

        for idx, seg_path in enumerate(state.segment_paths):
            if idx > 0 and idx - 1 < len(state.joint_paths):
                joint_path = state.joint_paths[idx - 1]
                targets = state.joint_drive_targets.get(joint_path, {})
                orientation = orientation * self._compose_joint_rotation(targets)

            forward = Gf.Rotation(orientation).TransformDir(Gf.Vec3d(1.0, 0.0, 0.0))
            seg_len = seg_lengths[idx] if idx < len(seg_lengths) else state.params.segment_length
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
        self._update_anchors(state)

        state.dirty = True

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

    def _define_anchor(self, stage, path: str):
        prim = stage.GetPrimAtPath(path)
        if prim and prim.IsValid():
            return prim
        return UsdGeom.Xform.Define(stage, Sdf.Path(path)).GetPrim()

    def _update_anchors(self, state: CableState):
        """Place start/end anchors at rope tips."""
        # Allow anchors to act as user-controlled handles when requested.
        if hasattr(state, "anchors_follow_rope") and not state.anchors_follow_rope:
            return

        stage = self._usd_context.get_stage()
        if not stage or not state.segment_paths:
            return

        seg_lengths = state.segment_lengths or [state.params.segment_length] * len(state.segment_paths)
        first_pose = self._segment_frame(stage, state.segment_paths[0])
        last_pose = self._segment_frame(stage, state.segment_paths[-1])
        if first_pose:
            pos, rot = first_pose
            dir_x = Gf.Rotation(rot).TransformDir(Gf.Vec3d(1.0, 0.0, 0.0))
            tip = self._prim_world_pos(stage, f"{state.segment_paths[0]}/tip/attach") or self._prim_world_pos(
                stage, f"{state.segment_paths[0]}/tip"
            )
            if tip is None:
                tip = pos - dir_x * (seg_lengths[0] * 0.5)
            self._set_world_transform(state.anchor_start, tip, rot)
        if last_pose:
            pos, rot = last_pose
            dir_x = Gf.Rotation(rot).TransformDir(Gf.Vec3d(1.0, 0.0, 0.0))
            tip = self._prim_world_pos(stage, f"{state.segment_paths[-1]}/tip/attach") or self._prim_world_pos(
                stage, f"{state.segment_paths[-1]}/tip"
            )
            if tip is None:
                tip = pos + dir_x * (seg_lengths[-1] * 0.5)
            self._set_world_transform(state.anchor_end, tip, rot)

    def _set_world_transform(self, path: str, pos: Gf.Vec3d, rot: Gf.Quatd):
        stage = self._usd_context.get_stage()
        if not stage:
            return
        prim = stage.GetPrimAtPath(path)
        if not prim or not prim.IsValid():
            return
        xf = UsdGeom.Xformable(prim)

        # Convert desired world-space transform into the local space of the parent
        # so anchors remain glued to rope tips even when the cable root moves.
        parent = prim.GetParent()
        parent_world = Gf.Matrix4d(1.0)
        if parent and parent.IsValid():
            parent_xf = UsdGeom.Xformable(parent)
            parent_world = parent_xf.ComputeLocalToWorldTransform(Usd.TimeCode.Default())

        inv_parent = parent_world.GetInverse()
        local_pos = inv_parent.Transform(pos)

        parent_rot = parent_world.ExtractRotation().GetQuat()
        local_rot = parent_rot.GetInverse() * rot

        xf.ClearXformOpOrder()
        xf.AddTranslateOp().Set(Gf.Vec3f(local_pos))
        qf = Gf.Quatf(float(local_rot.GetReal()), Gf.Vec3f(local_rot.GetImaginary()))
        xf.AddOrientOp().Set(qf)


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

    def _move_anchor_to_world(self, stage, anchor_path: str):
        """Move anchor prim out of the cable root to /World while keeping its name unique."""
        if not anchor_path:
            return
        prim = stage.GetPrimAtPath(anchor_path)
        if not prim or not prim.IsValid():
            return

        # If already under /World, nothing to do.
        parent = prim.GetParent()
        if parent and parent.GetPath() == Sdf.Path("/World"):
            return

        dest_path = self._make_unique_world_child(stage, Sdf.Path(anchor_path).name)
        try:
            stage.MovePrim(anchor_path, dest_path)
            carb.log_info(f"[RopeBuilder] Moved anchor {anchor_path} to {dest_path}.")
        except Exception as exc:
            carb.log_warn(f"[RopeBuilder] Failed to move anchor {anchor_path}: {exc}")

    def _make_unique_root_path(self, stage, base_name: str) -> str:
        """Return a unique root path under /World avoiding collisions with existing cables."""
        base = re.sub(r"[^\w]", "_", base_name) or "cable"
        candidate = f"/World/{base}"
        suffix = 2
        while candidate in self._cables or stage.GetPrimAtPath(candidate).IsValid():
            candidate = f"/World/{base}_{suffix:02d}"
            suffix += 1
        return candidate

    def _make_unique_world_child(self, stage, base_name: str) -> str:
        """Return a unique child path under /World for rehoming anchors."""
        base = re.sub(r"[^\w]", "_", base_name) or "anchor"
        candidate = f"/World/{base}"
        suffix = 2
        while stage.GetPrimAtPath(candidate).IsValid():
            candidate = f"/World/{base}_{suffix:02d}"
            suffix += 1
        return candidate

    def _segment_plan(self, params: RopeParameters) -> List[Tuple[str, float]]:
        """Return ordered list of (name, length) including short start/end segments."""
        base_len = params.segment_length
        short_len = base_len * 0.25
        plan: List[Tuple[str, float]] = [("segment_start", short_len)]
        for i in range(params.segment_count):
            plan.append((f"segment_{i:02d}", base_len))
        plan.append(("segment_end", short_len))
        return plan

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
    ) -> Tuple[RopeParameters, Dict[str, Dict[str, Tuple[float, float]]], List[float]]:
        radius = 0.01
        seg_lengths: List[float] = []
        mass_total = 1.0
        limits: Dict[str, Dict[str, Tuple[float, float]]] = {}

        # Radius + individual segment lengths from collisions or spacing.
        for i, spath in enumerate(segment_paths):
            rad = radius
            length = None
            col_prim = stage.GetPrimAtPath(f"{spath}/collision")
            if col_prim and col_prim.IsValid():
                rad_attr = col_prim.GetAttribute("radius")
                if rad_attr and rad_attr.HasAuthoredValueOpinion():
                    rad = float(rad_attr.Get())
                h_attr = col_prim.GetAttribute("height")
                if h_attr and h_attr.HasAuthoredValueOpinion():
                    length = float(h_attr.Get()) + 2.0 * rad
            if i == 0:
                radius = rad
            if length is None:
                if i + 1 < len(segment_paths):
                    p0 = self._segment_world_pos(stage, spath)
                    p1 = self._segment_world_pos(stage, segment_paths[i + 1])
                    if p0 is not None and p1 is not None:
                        length = float((Gf.Vec3d(p1) - Gf.Vec3d(p0)).GetLength())
            if length is None:
                length = 0.1
            seg_lengths.append(length)

        total_len = sum(seg_lengths) if seg_lengths else 0.1

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

        inner_count = max(len(segment_paths) - 2, 1)
        span = DEFAULT_PARAMS.rot_limit_span
        if joint_paths:
            x_low, x_high = limits.get(joint_paths[0], {}).get("rotX", (-30.0, 30.0))
            span = float(x_high - x_low)
        params = RopeParameters(
            length=total_len,
            radius=radius,
            segment_count=inner_count,
            mass=mass_total,
            rot_limit_span=span,
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
        half_span = max(span * 0.5, 0.0)
        params.rot_x_low = params.rot_y_low = params.rot_z_low = -half_span
        params.rot_x_high = params.rot_y_high = params.rot_z_high = half_span
        return params, limits, seg_lengths

    def get_joint_local_offsets(self, root_path: Optional[str] = None) -> List[Dict[str, Any]]:
        """Return local offset data (as shown in Isaac Sim D6 Joint Properties) for active or given cable."""
        state = self._get_state(root_path, require=False)
        if not state:
            return []
        out: List[Dict[str, Any]] = []
        for idx, jp in enumerate(state.joint_paths):
            offsets = state.joint_local_offsets.get(jp, {})
            out.append({
                "index": idx,
                "path": jp,
                **offsets,
            })
        return out

    def _infer_joint_targets_from_pose(
        self, stage, segment_paths: List[str], joint_paths: List[str]
    ) -> Dict[str, Dict[str, float]]:
        """Infer rotX/Y/Z targets from current world orientations of neighboring segments."""
        targets: Dict[str, Dict[str, float]] = {}
        for idx, jp in enumerate(joint_paths):
            rot_xyz = (0.0, 0.0, 0.0)
            if idx < len(segment_paths) - 1:
                pose0 = self._segment_frame(stage, segment_paths[idx])
                pose1 = self._segment_frame(stage, segment_paths[idx + 1])
                if pose0 and pose1:
                    qrel = pose0[1].GetInverse() * pose1[1]
                    rot_xyz = self._quat_to_euler_xyz_deg(qrel)

            targets[jp] = {
                "rotX": float(rot_xyz[0]),
                "rotY": float(rot_xyz[1]),
                "rotZ": float(rot_xyz[2]),
            }
        return targets

    @staticmethod
    def _quat_to_euler_xyz_deg(q) -> Tuple[float, float, float]:
        """Convert a quaternion (Gf.Quatf/Quatd) to XYZ Euler angles in degrees."""
        try:
            # Normalize to Quatd for stability
            if isinstance(q, Gf.Quatf):
                qd = Gf.Quatd(float(q.GetReal()), Gf.Vec3d(q.GetImaginary()))
            else:
                qd = q
            r = Gf.Rotation(qd)
            x_axis = Gf.Vec3d(1.0, 0.0, 0.0)
            y_axis = Gf.Vec3d(0.0, 1.0, 0.0)
            z_axis = Gf.Vec3d(0.0, 0.0, 1.0)
            rx, ry, rz = r.Decompose(x_axis, y_axis, z_axis)
            return float(rx), float(ry), float(rz)
        except Exception:
            return 0.0, 0.0, 0.0
