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
import carb
import omni.usd


@dataclass
class RopeParameters:
    """Container describing the rope layout and physical properties."""

    length: float = 2.0  # meters
    diameter: float = 0.05  # meters
    capsule_count: int = 10
    mass: float = 1.0  # kilograms
    joint_stiffness: float = 500.0  # N*m/rad equivalent (placeholder)
    joint_damping: float = 5.0  # N*m*s/rad equivalent (placeholder)

    @property
    def capsule_length(self) -> float:
        if self.capsule_count <= 0:
            return 0.0
        return self.length / self.capsule_count

    @property
    def capsule_mass(self) -> float:
        if self.capsule_count <= 0:
            return 0.0
        return self.mass / self.capsule_count


class RopeBuilderController:
    """Owns the USD prims that make up the rope and implements the create/delete logic.

    The actual PhysX authoring will be added iteratively; for now the class tracks the
    requested parameters and provides a place to hook stage operations into.
    """

    def __init__(self):
        self._usd_context = omni.usd.get_context()
        self._params = RopeParameters()
        self._rope_root_path = "/RopeBuilder/Rope"
        self._rope_exists = False

    @property
    def parameters(self) -> RopeParameters:
        return self._params

    def set_parameters(self, params: RopeParameters):
        self._params = params
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

        # Placeholder: actual prim authoring will be implemented in subsequent steps.
        carb.log_info(
            "[RopeBuilder] Requested rope creation "
            f"(length={self._params.length} m, diameter={self._params.diameter} m, "
            f"capsules={self._params.capsule_count}, mass={self._params.mass} kg)."
        )
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

    def rope_exists(self) -> bool:
        return self._rope_exists

    def validate_parameters(self) -> bool:
        return self._validate_params(self._params)

    @staticmethod
    def _validate_params(params: RopeParameters) -> bool:
        """Basic sanity checks to catch common input mistakes."""
        return all(
            [
                params.length > 0.0,
                params.diameter > 0.0,
                params.capsule_count > 1,
                params.mass > 0.0,
                params.joint_stiffness >= 0.0,
                params.joint_damping >= 0.0,
            ]
        )
