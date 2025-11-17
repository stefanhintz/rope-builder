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

import omni.ui as ui
from isaacsim.gui.components.element_wrappers import CollapsableFrame
from isaacsim.gui.components.ui_utils import get_style
from omni.usd import StageEventType

from .scenario import RopeBuilderController, RopeParameters


class UIBuilder:
    """Creates the layout for the Rope Builder control window."""

    def __init__(self):
        self.frames = []
        self.wrapped_ui_elements = []

        self._controller = RopeBuilderController()
        self._status_model = ui.SimpleStringModel("No rope created.")
        self._param_models = {}
        self._param_constraints = {}
        self._syncing_models = False

    ###################################################################################
    #           The Functions Below Are Called Automatically By extension.py
    ###################################################################################

    def on_menu_callback(self):
        pass

    def on_timeline_event(self, event):
        # Rope authoring does not depend on the timeline yet.
        pass

    def on_physics_step(self, step: float):
        pass

    def on_stage_event(self, event):
        if event.type == int(StageEventType.OPENED):
            self._controller.delete_rope()
            self._reset_ui()

    def cleanup(self):
        self._controller.delete_rope()

    def build_ui(self):
        """Called when the window is (re)built."""
        self._param_models = {}
        self._param_constraints = {}

        with CollapsableFrame("Rope Parameters", collapsed=False):
            with ui.VStack(style=get_style(), spacing=8, height=0):
                self._build_float_field("Length (m)", "length", min_value=0.1, step=0.05)
                self._build_float_field("Diameter (m)", "diameter", min_value=0.005, step=0.005)
                self._build_int_field("Segments", "segment_count", min_value=2, step=1)
                self._build_float_field("Mass (kg)", "mass", min_value=0.01, step=0.1)
                self._build_float_field("Joint Stiffness", "joint_stiffness", min_value=0.0, step=10.0)
                self._build_float_field("Joint Damping", "joint_damping", min_value=0.0, step=1.0)

        with CollapsableFrame("Actions", collapsed=False):
            with ui.VStack(style=get_style(), spacing=8, height=0):
                self._create_btn = ui.Button("Create Rope", clicked_fn=self._on_create_rope)
                self._delete_btn = ui.Button(
                    "Delete Rope", clicked_fn=self._on_delete_rope, enabled=self._controller.rope_exists()
                )
                ui.Label("Status:", style=get_style())
                ui.Label("", word_wrap=True, model=self._status_model)

        self._reset_ui()

    ###################################################################################
    #                             UI CALLBACKS AND HELPERS
    ###################################################################################

    def _build_float_field(self, label: str, param_key: str, min_value: float, step: float):
        params = self._controller.parameters
        model = ui.SimpleFloatModel(getattr(params, param_key))
        model.add_value_changed_fn(lambda m, key=param_key: self._on_param_change(key, m.as_float))

        self._param_constraints[param_key] = {"min": min_value, "type": float}

        with ui.HStack(height=0):
            ui.Label(label, width=120, style=get_style())
            ui.FloatField(model=model)

        self._param_models[param_key] = model

    def _build_int_field(self, label: str, param_key: str, min_value: int, step: int):
        params = self._controller.parameters
        model = ui.SimpleIntModel(getattr(params, param_key))
        model.add_value_changed_fn(lambda m, key=param_key: self._on_param_change(key, m.as_int))

        self._param_constraints[param_key] = {"min": min_value, "type": int}

        with ui.HStack(height=0):
            ui.Label(label, width=120, style=get_style())
            ui.IntField(model=model)

        self._param_models[param_key] = model

    def _on_param_change(self, key: str, value):
        if self._syncing_models:
            return

        value, changed = self._apply_constraints(key, value)
        model = self._param_models.get(key)

        params = self._controller.parameters
        setattr(params, key, value)
        self._controller.set_parameters(params)
        self._update_status(f"Updated {key.replace('_', ' ')}.", warn=not self._controller.validate_parameters())

        if changed and model is not None:
            self._syncing_models = True
            try:
                if isinstance(model, ui.SimpleFloatModel):
                    model.set_value(value)
                else:
                    model.set_value(int(value))
            finally:
                self._syncing_models = False

    def _on_create_rope(self):
        try:
            prim_path = self._controller.create_rope()
        except (RuntimeError, ValueError) as exc:
            self._update_status(str(exc), warn=True)
            return

        self._delete_btn.enabled = True
        self._update_status(f"Rope prims initialized at {prim_path}.", warn=False)

    def _on_delete_rope(self):
        self._controller.delete_rope()
        self._delete_btn.enabled = False
        self._update_status("Rope deleted.", warn=False)

    def _reset_ui(self):
        params = self._controller.parameters
        for key, model in self._param_models.items():
            value = getattr(params, key)
            if isinstance(model, ui.SimpleFloatModel):
                model.set_value(value)
            else:
                model.set_value(int(value))

        if hasattr(self, "_delete_btn"):
            self._delete_btn.enabled = self._controller.rope_exists()

        self._update_status("Ready to create a rope.", warn=False)

    def _update_status(self, message: str, warn: bool):
        prefix = "Warning: " if warn else ""
        self._status_model.set_value(f"{prefix}{message}")

    def _apply_constraints(self, key: str, value):
        constraint = self._param_constraints.get(key)
        if not constraint:
            return value, False

        min_value = constraint.get("min")
        updated_value = value
        if min_value is not None:
            updated_value = max(updated_value, min_value)

        if constraint.get("type") is int:
            updated_value = int(round(updated_value))

        return updated_value, updated_value != value
