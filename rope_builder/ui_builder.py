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

import math

import omni.ui as ui
from omni.ui import color as cl
from isaacsim.gui.components.element_wrappers import CollapsableFrame
from isaacsim.gui.components.ui_utils import get_style
from omni.usd import StageEventType

from .scenario import ROT_AXES, RopeBuilderController


class UIBuilder:
    """Creates the layout for the Rope Builder control window."""

    def __init__(self):
        self.frames = []
        self.wrapped_ui_elements = []

        self._controller = RopeBuilderController()
        self._status_model = ui.SimpleStringModel("No cable created.")
        self._param_models = {}
        self._param_constraints = {}
        self._syncing_models = False
        self._segment_slider = None
        self._subscription_btn = None
        self._reset_joint_btn = None
        self._joint_frame = None
        self._joint_slider_models = {}
        self._toggle_vis_btn = None
        self._cable_name_model = ui.SimpleStringModel("cable")
        self._import_path_model = ui.SimpleStringModel("/World/cable")
        self._active_path_model = ui.SimpleStringModel("")
        self._known_cables_model = ui.SimpleStringModel("No cables yet.")
        self._plug_start_model = ui.SimpleStringModel("")
        self._plug_end_model = ui.SimpleStringModel("")

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
            # Only clear cached state on stage reopen; keep existing prims intact.
            self._controller.forget_all_cables()
            self._reset_ui()

    def cleanup(self):
        # Do not delete prims on shutdown; just clear controller state/subscriptions.
        self._controller.forget_all_cables()

    def build_ui(self):
        """Called when the window is (re)built."""
        self._param_models = {}
        self._param_constraints = {}

        with CollapsableFrame("Cable Parameters", collapsed=False):
            with ui.VStack(style=get_style(), spacing=8, height=0):
                self._build_float_field("Length (m)", "length", min_value=0.1, step=0.05)
                self._build_float_field("Radius (m)", "radius", min_value=0.001, step=0.001)
                self._build_float_field("Spline extend (m)", "curve_extension", min_value=0.0, step=0.005)
                self._build_segment_slider("Segments", "segment_count", min_value=2)
                self._build_float_field("Total Mass (kg)", "mass", min_value=0.01, step=0.05)

        with CollapsableFrame("Joint Limits (degrees)", collapsed=False):
            with ui.VStack(style=get_style(), spacing=8, height=0):
                self._build_float_field("rotX low", "rot_x_low", min_value=-180.0, step=1.0)
                self._build_float_field("rotX high", "rot_x_high", min_value=-180.0, step=1.0)
                self._build_float_field("rotY low", "rot_y_low", min_value=-180.0, step=1.0)
                self._build_float_field("rotY high", "rot_y_high", min_value=-180.0, step=1.0)
                self._build_float_field("rotZ low", "rot_z_low", min_value=-180.0, step=1.0)
                self._build_float_field("rotZ high", "rot_z_high", min_value=-180.0, step=1.0)

        with CollapsableFrame("Drive Settings", collapsed=False):
            with ui.VStack(style=get_style(), spacing=8, height=0):
                self._build_float_field("Stiffness", "drive_stiffness", min_value=0.0, step=10.0)
                self._build_float_field("Damping", "drive_damping", min_value=0.0, step=1.0)
                self._build_float_field("Max Force", "drive_max_force", min_value=0.0, step=10.0)

        with CollapsableFrame("Actions", collapsed=False):
            with ui.VStack(style=get_style(), spacing=8, height=0):
                with ui.HStack(height=0):
                    ui.Label("Cable name", width=140, style=get_style())
                    ui.StringField(model=self._cable_name_model)
                with ui.HStack(height=0):
                    ui.Label("Import root path", width=140, style=get_style())
                    ui.StringField(model=self._import_path_model)
                    ui.Button("Import cable", clicked_fn=self._on_import_rope)
                with ui.HStack(height=0):
                    ui.Label("Active cable", width=140, style=get_style())
                    ui.StringField(model=self._active_path_model)
                    ui.Button("Set active", clicked_fn=self._on_set_active_cable)
                ui.Label("", word_wrap=True, model=self._known_cables_model)
                with ui.HStack(height=0):
                    ui.Label("Plug start path", width=140, style=get_style())
                    ui.StringField(model=self._plug_start_model)
                with ui.HStack(height=0):
                    ui.Label("Plug end path", width=140, style=get_style())
                    ui.StringField(model=self._plug_end_model)
                    ui.Button("Attach plugs", clicked_fn=self._on_attach_plugs)
                    ui.Button("Discover plugs", clicked_fn=self._on_discover_plugs)
                self._create_btn = ui.Button("Create Cable", clicked_fn=self._on_create_rope)
                self._delete_btn = ui.Button(
                    "Delete Cable", clicked_fn=self._on_delete_rope, enabled=self._controller.rope_exists()
                )
                self._subscription_btn = ui.Button(
                    "Subscribe spline update", clicked_fn=self._on_toggle_subscription, enabled=False
                )
                self._reset_joint_btn = ui.Button(
                    "Reset joint targets", clicked_fn=self._on_reset_joints, enabled=False
                )
                self._toggle_vis_btn = ui.Button(
                    "Show collisions", clicked_fn=self._on_toggle_visibility, enabled=False
                )
                ui.Label("Status:", style=get_style())
                ui.Label("", word_wrap=True, model=self._status_model)

        with CollapsableFrame("Joint Controls", collapsed=False):
            self._joint_frame = ui.Frame()
            with self._joint_frame:
                ui.Label("Create a cable to edit joint drive targets.", style=get_style())

        self._reset_ui()

    ###################################################################################
    #                             UI CALLBACKS AND HELPERS
    ###################################################################################

    def _build_float_field(self, label: str, param_key: str, min_value: float, step: float, max_value: float = None):
        params = self._controller.parameters
        model = ui.SimpleFloatModel(getattr(params, param_key))
        model.add_value_changed_fn(lambda m, key=param_key: self._on_param_change(key, m.as_float))

        self._param_constraints[param_key] = {"min": min_value, "type": float, "max": max_value}

        with ui.HStack(height=0):
            ui.Label(label, width=140, style=get_style())
            ui.FloatField(model=model)

        self._param_models[param_key] = model

    def _build_int_field(self, label: str, param_key: str, min_value: int, step: int):
        params = self._controller.parameters
        model = ui.SimpleIntModel(getattr(params, param_key))
        model.add_value_changed_fn(lambda m, key=param_key: self._on_param_change(key, m.as_int))

        self._param_constraints[param_key] = {"min": min_value, "type": int}

        with ui.HStack(height=0):
            ui.Label(label, width=140, style=get_style())
            ui.IntField(model=model)

        self._param_models[param_key] = model

    def _build_segment_slider(self, label: str, param_key: str, min_value: int):
        params = self._controller.parameters
        model = ui.SimpleIntModel(getattr(params, param_key))
        model.add_value_changed_fn(lambda m, key=param_key: self._on_param_change(key, m.as_int))

        self._param_constraints[param_key] = {"min": min_value, "type": int, "max_fn": self._segment_max_limit}

        with ui.HStack(height=0):
            ui.Label(label, width=140, style=get_style())
            self._segment_slider = ui.IntSlider(model=model, min=min_value, max=self._segment_max_limit())

        self._param_models[param_key] = model

    def _on_param_change(self, key: str, value):
        if self._syncing_models:
            return

        value, changed = self._apply_constraints(key, value)
        model = self._param_models.get(key)

        params = self._controller.parameters
        setattr(params, key, value)
        self._controller.set_parameters(params)
        status_msg = f"Updated {key.replace('_', ' ')}."
        self._update_status(status_msg, warn=not self._controller.validate_parameters())

        if key in {"length", "radius"}:
            self._refresh_segment_slider_limit(clamp_value=True)

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
            name = self._model_string(self._cable_name_model, "cable")
            prim_path = self._controller.create_rope(name)
        except (RuntimeError, ValueError) as exc:
            self._update_status(str(exc), warn=True)
            return

        self._active_path_model.set_value(prim_path)
        self._refresh_known_cables_label()
        self._delete_btn.enabled = True
        self._reset_joint_btn.enabled = True
        self._subscription_btn.enabled = True
        self._toggle_vis_btn.enabled = True
        self._refresh_subscription_btn()
        self._refresh_visibility_btn()
        self._build_joint_controls()
        self._update_status(f"Cable prims initialized at {prim_path}.", warn=False)
        self._auto_attach_plugs()

    def _on_delete_rope(self):
        self._controller.delete_rope()
        active_exists = self._controller.rope_exists()
        self._delete_btn.enabled = active_exists
        self._reset_joint_btn.enabled = active_exists
        self._subscription_btn.enabled = active_exists
        self._toggle_vis_btn.enabled = active_exists
        self._active_path_model.set_value(self._controller.active_cable_path() or "")
        self._refresh_known_cables_label()
        self._refresh_subscription_btn()
        self._refresh_visibility_btn()
        self._clear_joint_controls()
        self._update_status("Cable deleted.", warn=False)

    def _on_import_rope(self):
        path = self._model_string(self._import_path_model, "/World/cable")
        try:
            prim_path = self._controller.import_cable(path)
        except (RuntimeError, ValueError) as exc:
            self._update_status(str(exc), warn=True)
            return
        self._active_path_model.set_value(prim_path)
        self._refresh_known_cables_label()
        self._delete_btn.enabled = True
        self._reset_joint_btn.enabled = True
        self._subscription_btn.enabled = True
        self._toggle_vis_btn.enabled = True
        self._refresh_subscription_btn()
        self._refresh_visibility_btn()
        self._build_joint_controls()
        self._update_status(f"Imported cable at {prim_path}.", warn=False)
        self._auto_attach_plugs()

    def _on_set_active_cable(self):
        path = self._model_string(self._active_path_model, "")
        if not path:
            self._update_status("Enter a cable root path to activate.", warn=True)
            return
        if not self._controller.set_active_cable(path):
            self._update_status(f"No known cable at {path}.", warn=True)
            return
        self._refresh_known_cables_label()
        self._delete_btn.enabled = True
        self._reset_joint_btn.enabled = True
        self._subscription_btn.enabled = True
        self._toggle_vis_btn.enabled = True
        self._refresh_subscription_btn()
        self._refresh_visibility_btn()
        self._build_joint_controls()
        plug_start, plug_end = self._controller.get_plug_paths()
        if hasattr(self._plug_start_model, "set_value"):
            self._plug_start_model.set_value(plug_start or "")
        if hasattr(self._plug_end_model, "set_value"):
            self._plug_end_model.set_value(plug_end or "")
        self._update_status(f"Active cable set to {path}.", warn=False)
        self._auto_attach_plugs()

    def _on_attach_plugs(self):
        start_path = self._model_string(self._plug_start_model, "")
        end_path = self._model_string(self._plug_end_model, "")
        try:
            self._controller.attach_plugs(start_path or None, end_path or None)
        except (RuntimeError, ValueError) as exc:
            self._update_status(str(exc), warn=True)
            return
        msg = "Recorded plug paths (no joints created automatically)."
        self._update_status(msg, warn=False)

    def _on_discover_plugs(self):
        try:
            start_path, end_path = self._controller.discover_plugs_from_joints()
        except (RuntimeError, ValueError) as exc:
            self._update_status(str(exc), warn=True)
            return
        if hasattr(self._plug_start_model, "set_value"):
            self._plug_start_model.set_value(start_path or "")
        if hasattr(self._plug_end_model, "set_value"):
            self._plug_end_model.set_value(end_path or "")
        if start_path or end_path:
            self._update_status(
                f"Discovered plugs: start={start_path or 'none'}, end={end_path or 'none'}.", warn=False
            )
        else:
            self._update_status("No joints found on start/end segments to infer plugs.", warn=True)

    def _model_string(self, model, default: str = "") -> str:
        if hasattr(model, "as_string"):
            return model.as_string
        if hasattr(model, "get_value_as_string"):
            return model.get_value_as_string()
        return default

    def _on_toggle_subscription(self):
        if not self._controller.rope_exists():
            self._update_status("Create a cable before subscribing.", warn=True)
            return

        try:
            if self._controller.curve_subscription_active():
                self._controller.stop_curve_updates()
                self._update_status("Stopped spline subscription.", warn=False)
            else:
                self._controller.start_curve_updates()
                self._update_status("Spline now updates from segment positions.", warn=False)
        except (RuntimeError, ValueError) as exc:
            self._update_status(str(exc), warn=True)

        self._refresh_subscription_btn()

    def _on_reset_joints(self):
        self._controller.reset_joint_drive_targets()
        data = {info.get("index"): info for info in self._controller.get_joint_control_data()}
        for (joint_idx, axis), model in self._joint_slider_models.items():
            limits = data.get(joint_idx, {}).get("limits", {})
            low, high = limits.get(axis, (-180.0, 180.0))
            if model and low <= 0.0 <= high:
                model.set_value(0.0)

    def _on_reset_joint_axis(self, joint_index: int, axis: str):
        """Reset a single axis on one joint to zero within limits."""
        data = {info.get("index"): info for info in self._controller.get_joint_control_data()}
        limits = data.get(joint_index, {}).get("limits", {})
        low, high = limits.get(axis, (-180.0, 180.0))
        if low <= 0.0 <= high:
            clamped = self._controller.set_joint_drive_target(joint_index, axis, 0.0, apply_pose=True)
            model = self._joint_slider_models.get((joint_index, axis))
            if model and abs(model.as_float - clamped) > 1e-6:
                model.set_value(clamped)

    def _on_toggle_visibility(self):
        if not self._controller.rope_exists():
            self._update_status("Create a cable before toggling visibility.", warn=True)
            return
        show_curve = self._controller.toggle_visibility()
        self._refresh_visibility_btn()
        msg = "Showing spline (collisions hidden)." if show_curve else "Showing collisions (spline hidden)."
        self._update_status(msg, warn=False)

    def _refresh_known_cables_label(self):
        paths = self._controller.list_cable_paths()
        if not paths:
            self._known_cables_model.set_value("No cables yet.")
        else:
            active = self._controller.active_cable_path()
            text = "Known cables: " + ", ".join([p + ("*" if p == active else "") for p in paths])
            self._known_cables_model.set_value(text)

    def _reset_ui(self):
        params = self._controller.parameters
        for key, model in self._param_models.items():
            value = getattr(params, key)
            if isinstance(model, ui.SimpleFloatModel):
                model.set_value(value)
            else:
                model.set_value(int(value))

        if hasattr(self, "_delete_btn"):
            exists = self._controller.rope_exists()
            self._delete_btn.enabled = exists
            self._reset_joint_btn.enabled = exists
            self._subscription_btn.enabled = exists
            self._toggle_vis_btn.enabled = exists
            if exists:
                self._auto_attach_plugs()

        plug_start, plug_end = self._controller.get_plug_paths()
        if hasattr(self._plug_start_model, "set_value"):
            self._plug_start_model.set_value(plug_start or "")
        if hasattr(self._plug_end_model, "set_value"):
            self._plug_end_model.set_value(plug_end or "")

        self._active_path_model.set_value(self._controller.active_cable_path() or "")
        self._refresh_known_cables_label()
        self._refresh_segment_slider_limit(clamp_value=True)
        self._refresh_subscription_btn()
        self._refresh_visibility_btn()
        self._build_joint_controls()
        self._update_status("Ready to create a cable.", warn=False)

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

        max_fn = constraint.get("max_fn")
        max_value = constraint.get("max")
        if max_fn:
            max_value = max_fn()

        if max_value is not None:
            updated_value = min(updated_value, max_value)

        return updated_value, updated_value != value

    def _segment_max_limit(self) -> int:
        """Compute the maximum segments allowed so each is longer than two radii."""
        params = self._controller.parameters
        if params.radius <= 0.0:
            return 2

        # Require each segment length to exceed two radii so the collider is valid.
        ratio = params.length / (params.radius * 2.0)
        max_segments = int(math.floor(ratio - 1e-6))
        return max(max_segments, 2)

    def _refresh_segment_slider_limit(self, clamp_value: bool = False):
        if not self._segment_slider:
            return

        max_segments = self._segment_max_limit()
        if hasattr(self._segment_slider, "max"):
            self._segment_slider.max = max_segments
        if hasattr(self._segment_slider, "min"):
            self._segment_slider.min = 2

        if clamp_value:
            model = self._param_models.get("segment_count")
            if model and model.as_int > max_segments:
                new_value = max_segments
                self._syncing_models = True
                try:
                    model.set_value(new_value)
                finally:
                    self._syncing_models = False
                self._on_param_change("segment_count", new_value)

    def _refresh_subscription_btn(self):
        if not self._subscription_btn:
            return
        subscribed = self._controller.curve_subscription_active()
        self._subscription_btn.text = "Unsubscribe spline update" if subscribed else "Subscribe spline update"
        self._subscription_btn.enabled = self._controller.rope_exists()

    def _refresh_visibility_btn(self):
        if not self._toggle_vis_btn:
            return
        show_curve = self._controller.showing_curve()
        self._toggle_vis_btn.text = "Show collisions" if show_curve else "Show spline"
        self._toggle_vis_btn.enabled = self._controller.rope_exists()

    def _auto_attach_plugs(self):
        # Auto-attach when paths are present and a cable exists.
        if not self._controller.rope_exists():
            return
        start_path = self._model_string(self._plug_start_model, "")
        end_path = self._model_string(self._plug_end_model, "")
        if not start_path and not end_path:
            return
        try:
            self._controller.attach_plugs(start_path or None, end_path or None)
        except (RuntimeError, ValueError):
            pass

    def _build_joint_controls(self):
        if not self._joint_frame:
            return

        self._joint_frame.clear()
        self._joint_slider_models = {}
        data = self._controller.get_joint_control_data()

        with self._joint_frame:
            if not data:
                ui.Label("Create a cable to edit joint drive targets.", style=get_style())
                return

            with ui.VStack(style=get_style(), spacing=6, height=0):
                for info in data:
                    idx = info.get("index", 0)
                    limits = info.get("limits", {})
                    targets = info.get("targets", {})
                    # Align label and sliders horizontally with compact spacing.
                    with ui.HStack(height=0, spacing=20):
                        ui.Label(f"Joint {idx}", width=80, style=get_style())
                        with ui.HStack(height=0, spacing=8):
                            for axis in ROT_AXES:
                                low, high = limits.get(axis, (-180.0, 180.0))
                                model = ui.SimpleFloatModel(targets.get(axis, 0.0))
                                model.add_value_changed_fn(
                                    lambda m, i=idx, ax=axis: self._on_joint_slider_changed(i, ax, m.as_float)
                                )
                                with ui.VStack(height=0, spacing=2):
                                    ui.Label(f"{axis}", width=24, style=get_style())
                                    with ui.HStack(height=18, spacing=4):
                                        ui.FloatSlider(min=low, max=high, model=model, style=get_style(), height=0)
                                        ui.Button(
                                            "",
                                            width=18,
                                            height=18,
                                            clicked_fn=lambda i=idx, ax=axis: self._on_reset_joint_axis(i, ax),
                                            tooltip=f"Reset {axis} to zero within limits",
                                            style={
                                                "Button": {
                                                    "background_color": cl(0xFF2196F3),
                                                    "border_color": cl(0xFF2196F3),
                                                    "border_width": 1,
                                                    "border_radius": 3,
                                                    "padding": 0,
                                                    "margin": 0,
                                                },
                                                "Button:hovered": {"background_color": cl(0xFF42A5F5)},
                                                "Button:pressed": {"background_color": cl(0xFF1E88E5)},
                                            },
                                        )
                                self._joint_slider_models[(idx, axis)] = model

    def _clear_joint_controls(self):
        if not self._joint_frame:
            return
        self._joint_frame.clear()
        self._joint_slider_models = {}
        with self._joint_frame:
            ui.Label("Create a cable to edit joint drive targets.", style=get_style())

    def _on_joint_slider_changed(self, joint_index: int, axis: str, value: float):
        clamped = self._controller.set_joint_drive_target(joint_index, axis, value, apply_pose=True)
        model = self._joint_slider_models.get((joint_index, axis))
        if model and abs(model.as_float - clamped) > 1e-6:
            model.set_value(clamped)
