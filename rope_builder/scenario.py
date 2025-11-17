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
from typing import List, Tuple

import carb
import omni.usd
from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdPhysics


@dataclass
class RopeParameters:
    """Container describing the rope layout and physical properties."""

    length: float = 2.0  # meters
    diameter: float = 0.05  # meters
    segment_count: int = 10
    mass: float = 1.0  # kilograms
    joint_stiffness: float = 500.0  # N*m/rad equivalent (placeholder)
    joint_damping: float = 5.0  # N*m*s/rad equivalent (placeholder)
    visual_radius_scale: float = 0.95
    visual_color_r: float = 0.8
    visual_color_g: float = 0.4
    visual_color_b: float = 0.1

    @property
    def segment_length(self) -> float:
        if self.segment_count <= 0:
            return 0.0
        return self.length / self.segment_count

    @property
    def segment_mass(self) -> float:
        if self.segment_count <= 0:
            return 0.0
        return self.mass / self.segment_count

    @property
    def visual_color(self) -> Tuple[float, float, float]:
        return (self.visual_color_r, self.visual_color_g, self.visual_color_b)


DEFAULT_PARAMS = RopeParameters()


class RopeBuilderController:
    """Owns the USD prims that make up the rope and implements the create/delete logic.

    The actual PhysX authoring will be added iteratively; for now the class tracks the
    requested parameters and provides a place to hook stage operations into.
    """

    def __init__(self):
        self._usd_context = omni.usd.get_context()
        self._params = RopeParameters()
        self._ensure_parameter_defaults()
        self._rope_root_path = "/RopeBuilder/Rope"
        self._rope_exists = False
        self._segment_paths: List[str] = []
        self._joint_paths: List[str] = []
        self._physics_scene_path = "/World/physicsScene"

    @property
    def parameters(self) -> RopeParameters:
        return self._params

    def set_parameters(self, params: RopeParameters):
        self._params = params
        self._ensure_parameter_defaults()
        carb.log_info(f"[RopeBuilder] Updated parameters: {self._params}")

    def create_rope(self) -> str:
        """Create the rope prim hierarchy on the current stage.

        For now this only validates and stores the intent so that the UI flow can be
        tested end-to-end.
        """
        if not self._validate_params(self._params):
            raise ValueError("Invalid rope parameters. Please fix the highlighted values.")

        stage = self._usd_context.get_stage()
        if stage is None:
            raise RuntimeError("No open USD stage. Create or open a stage before building a rope.")

        self.delete_rope()

        carb.log_info(
            "[RopeBuilder] Building rope with "
            f"{self._params.segment_count} segments, total length {self._params.length} m."
        )

        self._ensure_physics_scene(stage)
        root_prim = UsdGeom.Xform.Define(stage, Sdf.Path(self._rope_root_path))
        root_prim.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.0))

        self._segment_paths = []
        self._joint_paths = []

        prev_segment_path = None
        for idx in range(self._params.segment_count):
            segment_path = self._create_segment(stage, idx)
            if prev_segment_path:
                joint_path = self._create_d6_joint(stage, idx - 1, prev_segment_path, segment_path)
                self._joint_paths.append(joint_path)
            self._segment_paths.append(segment_path)
            prev_segment_path = segment_path

        self._rope_exists = True
        return self._rope_root_path

    def delete_rope(self):
        """Remove the rope prims from the stage if they exist."""
        if not self._rope_exists:
            return

        stage = self._usd_context.get_stage()
        if stage:
            prim = stage.GetPrimAtPath(self._rope_root_path)
            if prim and prim.IsValid():
                stage.RemovePrim(self._rope_root_path)
                carb.log_info("[RopeBuilder] Deleted rope prim hierarchy.")

        self._rope_exists = False
        self._segment_paths = []
        self._joint_paths = []

    def rope_exists(self) -> bool:
        return self._rope_exists

    def validate_parameters(self) -> bool:
        return self._validate_params(self._params)

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
                params.diameter > 0.0,
                params.segment_count > 1,
                params.mass > 0.0,
                params.joint_stiffness >= 0.0,
                params.joint_damping >= 0.0,
                segment_length > params.diameter,
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
        """Author the prims for an individual rope segment."""
        segment_path = Sdf.Path(f"{self._rope_root_path}/segment_{index:03d}")
        xform = UsdGeom.Xform.Define(stage, segment_path)

        center = self._segment_center_position(index)
        xform.AddTranslateOp().Set(center)

        radius = self._params.diameter * 0.5
        collision_path = segment_path.AppendPath("collision")
        collision = UsdGeom.Capsule.Define(stage, collision_path)
        collision.CreateRadiusAttr(radius)
        collision.CreateHeightAttr(self._capsule_height(radius))
        collision.CreateAxisAttr(UsdGeom.Tokens.x)
        UsdPhysics.CollisionAPI.Apply(collision.GetPrim())
        UsdGeom.Imageable(collision.GetPrim()).MakeInvisible()

        visual_path = segment_path.AppendPath("visual")
        visual = UsdGeom.Cylinder.Define(stage, visual_path)
        visual_radius = max(radius * self._params.visual_radius_scale, 1e-4)
        visual.CreateRadiusAttr(visual_radius)
        visual.CreateHeightAttr(self._params.segment_length)
        visual.CreateAxisAttr(UsdGeom.Tokens.x)
        visual.CreateDisplayColorAttr([self._params.visual_color])

        rigid_prim = xform.GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(rigid_prim)
        mass_api = UsdPhysics.MassAPI.Apply(rigid_prim)
        mass_api.CreateMassAttr(self._params.segment_mass)

        # Enable CCD for better stability when the rope moves quickly.
        physx_body = PhysxSchema.PhysxRigidBodyAPI.Apply(rigid_prim)
        if physx_body:
            physx_body.CreateEnableCCDAttr(True)

        return str(segment_path)

    def _create_d6_joint(self, stage, joint_index: int, body0_path: str, body1_path: str) -> str:
        """Create a D6 joint connecting two neighboring segments."""
        joint_path = Sdf.Path(f"{self._rope_root_path}/joint_{joint_index:03d}")
        joint_prim = stage.DefinePrim(joint_path, "PhysicsJoint")
        joint = UsdPhysics.Joint(joint_prim)
        joint.CreateBody0Rel().SetTargets([body0_path])
        joint.CreateBody1Rel().SetTargets([body1_path])

        half_length = self._params.segment_length * 0.5
        joint.CreateLocalPos0Attr().Set(Gf.Vec3f(half_length, 0.0, 0.0))
        joint.CreateLocalPos1Attr().Set(Gf.Vec3f(-half_length, 0.0, 0.0))

        PhysxSchema.PhysxJointAPI.Apply(joint_prim)

        drive_axes = [
            UsdPhysics.Tokens.transX,
            UsdPhysics.Tokens.transY,
            UsdPhysics.Tokens.transZ,
            UsdPhysics.Tokens.rotX,
            UsdPhysics.Tokens.rotY,
            UsdPhysics.Tokens.rotZ,
        ]
        for axis in drive_axes:
            drive = UsdPhysics.DriveAPI.Apply(joint_prim, axis)
            drive.CreateTargetPositionAttr().Set(0.0)
            drive.CreateTargetVelocityAttr().Set(0.0)
            drive.CreateStiffnessAttr(self._params.joint_stiffness)
            drive.CreateDampingAttr(self._params.joint_damping)
            drive.CreateMaxForceAttr(1e6)

        return str(joint_path)

    def _segment_center_position(self, index: int) -> Gf.Vec3d:
        """Compute the world-space center of the segment assuming a straight rope along X."""
        spacing = self._params.segment_length
        start = -0.5 * self._params.length + 0.5 * spacing
        x = start + index * spacing
        return Gf.Vec3d(x, 0.0, 0.0)

    def _capsule_height(self, radius: float) -> float:
        """Return the cylindrical height so total length matches the desired segment length."""
        total = self._params.segment_length
        return max(total - 2.0 * radius, 1e-4)
