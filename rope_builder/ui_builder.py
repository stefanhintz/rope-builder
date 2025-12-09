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

class _CableItem(ui.AbstractItem):
    def __init__(self, label: str):
        super().__init__()
        self.label = label
        self.model = ui.SimpleStringModel(label)


class _CableTreeModel(ui.AbstractItemModel):
    """Single-column flat model for cable root paths."""

    def __init__(self):
        super().__init__()
        self._items: list[_CableItem] = []

    def set_paths(self, paths: list[str]):
        self._items = [_CableItem(p) for p in paths]
        self._item_changed(None)  # notify view

    def get_item_value_model_count(self, item) -> int:
        # One text column.
        return 1

    def get_item_children(self, item):
        if item is None:
            return self._items
        return []

    def get_item_value_model(self, item, column_id=0):
        if item is None:
            return None
        return item.model

    def get_item_value(self, item, column_id=0):
        return item.label if item else ""

    def set_item_value(self, item, value, column_id=0):
        # read-only
        return False

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
        self._active_path_model = ui.SimpleStringModel("")
        # TreeView for discovered/known cables
        self._active_tree_model = _CableTreeModel()
        self._active_tree_view = None
        self._syncing_active_selection = False
        self._known_cables_model = ui.SimpleStringModel("No cables yet.")
        self._plug_start_model = ui.SimpleStringModel("")
        self._plug_end_model = ui.SimpleStringModel("")
        self._syncing_joint_build = False
        self._joint_limit_hint_model = ui.SimpleStringModel("")

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

        # ------------------------------------------------------------------
        # 1) Cable creation / authoring controls
        # ------------------------------------------------------------------
        with CollapsableFrame("Cable Creation", collapsed=False):
            with ui.VStack(style=get_style(), spacing=8, height=0):
                ui.Label("Cable Parameters", style=get_style())
                self._build_float_field("Length (m)", "length", min_value=0.1, step=0.05)
                self._build_float_field("Radius (m)", "radius", min_value=0.001, step=0.001)
                self._build_float_field("Spline extend (m)", "curve_extension", min_value=0.0, step=0.005)
                self._build_segment_slider("Segments", "segment_count", min_value=2)
                self._build_float_field("Total Mass (kg)", "mass", min_value=0.01, step=0.05)

                ui.Separator(height=6)
                ui.Label("Joint Limits (degrees)", style=get_style())
                self._build_joint_limit_span_field()

                ui.Separator(height=6)
                ui.Label("Drive Settings", style=get_style())
                self._build_float_field("Stiffness", "drive_stiffness", min_value=0.0, step=10.0)
                self._build_float_field("Damping", "drive_damping", min_value=0.0, step=1.0)
                self._build_float_field("Max Force", "drive_max_force", min_value=0.0, step=10.0)

                ui.Separator(height=6)
                with ui.HStack(height=0):
                    ui.Label("Cable name", width=140, style=get_style())
                    ui.StringField(model=self._cable_name_model)

                with ui.HStack(height=0, spacing=8):
                    self._create_btn = ui.Button("Create Cable", clicked_fn=self._on_create_rope)
                    self._delete_btn = ui.Button(
                        "Delete Cable", clicked_fn=self._on_delete_rope, enabled=self._controller.rope_exists()
                    )

        # ------------------------------------------------------------------
        # 2) Cable list / discovery
        # ------------------------------------------------------------------
        with CollapsableFrame("Cables", collapsed=False):
            with ui.VStack(style=get_style(), spacing=6, height=0):
                ui.Button("Discover cables", clicked_fn=self._on_discover_cables_button)
                self._active_tree_view = ui.TreeView(
                    self._active_tree_model,
                    root_visible=False,
                    header_visible=False,
                    columns_resizable=False,
                    column_widths=[ui.Length(280)],
                    height=140,
                    selection_changed_fn=self._on_active_tree_changed,
                )
                ui.Label("", word_wrap=True, model=self._known_cables_model)

        # ------------------------------------------------------------------
        # 3) Active cable controls (plugs + joint drive targets)
        # ------------------------------------------------------------------
        with CollapsableFrame("Active Cable Controls", collapsed=False):
            with ui.VStack(style=get_style(), spacing=8, height=0):
                ui.Label("Active cable", style=get_style())
                ui.StringField(model=self._active_path_model)

                with ui.HStack(height=0):
                    ui.Label("Plug start path", width=140, style=get_style())
                    ui.StringField(model=self._plug_start_model)
                with ui.HStack(height=0, spacing=8):
                    ui.Label("Plug end path", width=140, style=get_style())
                    ui.StringField(model=self._plug_end_model)
                    ui.Button("Discover plugs", clicked_fn=self._on_discover_plugs)

                with ui.HStack(height=0, spacing=8):
                    # Global spline subscribe/unsubscribe for all cables.
                    self._subscription_btn = ui.Button(
                        "Subscribe splines (all)", clicked_fn=self._on_sync_all_splines_button, enabled=False
                    )
                    self._toggle_vis_btn = ui.Button(
                        "Show collisions (all)", clicked_fn=self._on_toggle_visibility, enabled=False
                    )
                    self._reset_joint_btn = ui.Button(
                        "Reset joint targets", clicked_fn=self._on_reset_joints, enabled=False
                    )

                ui.Label("Joint drive targets", style=get_style())
                self._joint_frame = ui.Frame()
                with self._joint_frame:
                    ui.Label("Select or create a cable to edit joint drive targets.", style=get_style())

                ui.Separator(height=6)
                ui.Label("Status:", style=get_style())
                ui.Label("", word_wrap=True, model=self._status_model)

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

    def _build_joint_limit_span_field(self):
        params = self._controller.parameters
        span = self._current_joint_limit_span(params)
        model = ui.SimpleFloatModel(span)
        model.add_value_changed_fn(lambda m: self._on_joint_limit_span_change(m.as_float))

        self._param_constraints["joint_limit_span"] = {"min": 0.0, "type": float, "max": 360.0}

        with ui.VStack(height=0, spacing=2):
            with ui.HStack(height=0):
                ui.Label("Max DoF (deg)", width=140, style=get_style())
                ui.FloatField(model=model)
            self._joint_limit_hint_model.set_value(self._joint_limit_hint_text(span))
            ui.Label("", model=self._joint_limit_hint_model, style=get_style())

        self._param_models["joint_limit_span"] = model

    def _on_discover_cables_button(self):
        print("[RopeBuilder UI] Discover button clicked")
        
        try:
            found = self._controller.discover_cables("/World")
        except Exception as exc:
            self._update_status(str(exc), warn=True)
            return

        if found:
            self._update_status(f"Discovered: {', '.join(found)}", warn=False)
        else:
            self._update_status("No cables discovered.", warn=True)

        self._refresh_known_cables_label()
        self._refresh_active_tree()
        self._active_path_model.set_value(self._controller.active_cable_path() or "")

        # Enable global actions as soon as we have any cables, even if none is "active" yet.
        has_cables = bool(self._controller.list_cable_paths())
        if self._subscription_btn:
            self._subscription_btn.enabled = has_cables
        if self._toggle_vis_btn:
            self._toggle_vis_btn.enabled = has_cables

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

    def _on_joint_limit_span_change(self, value: float):
        if self._syncing_models:
            return

        span, changed = self._apply_constraints("joint_limit_span", value)
        params = self._controller.parameters
        params.rot_limit_span = span
        self._controller.set_parameters(params)

        hint_text = self._joint_limit_hint_text(span)
        self._joint_limit_hint_model.set_value(hint_text)
        status_msg = f"Updated joint limits to +/-{span * 0.5:.1f} degrees for rotX/rotY/rotZ."
        self._update_status(status_msg, warn=not self._controller.validate_parameters())

        model = self._param_models.get("joint_limit_span")
        if changed and model is not None:
            self._syncing_models = True
            try:
                model.set_value(span)
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
        self._refresh_active_tree()
        self._delete_btn.enabled = True
        self._reset_joint_btn.enabled = True
        self._subscription_btn.enabled = True
        self._toggle_vis_btn.enabled = True
        self._refresh_subscription_btn()
        self._refresh_visibility_btn()
        self._build_joint_controls()
        self._update_status(f"Cable prims initialized at {prim_path}.", warn=False)

    def _on_delete_rope(self):
        self._controller.delete_rope()
        active_exists = self._controller.rope_exists()
        self._delete_btn.enabled = active_exists
        self._reset_joint_btn.enabled = active_exists
        self._subscription_btn.enabled = active_exists
        self._toggle_vis_btn.enabled = active_exists
        self._active_path_model.set_value(self._controller.active_cable_path() or "")
        self._refresh_known_cables_label()
        self._refresh_active_tree()
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
        self._refresh_active_tree()
        self._delete_btn.enabled = True
        self._reset_joint_btn.enabled = True
        self._subscription_btn.enabled = True
        self._toggle_vis_btn.enabled = True
        self._refresh_subscription_btn()
        self._refresh_visibility_btn()
        self._build_joint_controls()
        self._update_status(f"Imported cable at {prim_path}.", warn=False)

    def _on_set_active_cable(self):
        path = self._model_string(self._active_path_model, "")
        if not path:
            self._update_status("Enter a cable root path to activate.", warn=True)
            return
        if not self._controller.set_active_cable(path):
            self._update_status(f"No known cable at {path}.", warn=True)
            return
        self._refresh_known_cables_label()
        self._refresh_active_tree()
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
        paths = self._controller.list_cable_paths()
        if not paths:
            self._update_status("Discover or create a cable before toggling visibility.", warn=True)
            return

        show_curve = self._controller.toggle_visibility_all()
        self._refresh_visibility_btn()
        msg = "Showing spline (collisions hidden) for all cables." if show_curve else \
              "Showing collisions (spline hidden) for all cables."
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

        self._joint_limit_hint_model.set_value(self._joint_limit_hint_text(self._current_joint_limit_span(params)))

        if hasattr(self, "_delete_btn"):
            # Active-cable-only actions depend on an active cable.
            active_exists = self._controller.rope_exists()
            self._delete_btn.enabled = active_exists
            self._reset_joint_btn.enabled = active_exists

            # Global actions depend on having any cables at all.
            has_cables = bool(self._controller.list_cable_paths())
            if self._subscription_btn:
                self._subscription_btn.enabled = has_cables
            if self._toggle_vis_btn:
                self._toggle_vis_btn.enabled = has_cables

        plug_start, plug_end = self._controller.get_plug_paths()
        if hasattr(self._plug_start_model, "set_value"):
            self._plug_start_model.set_value(plug_start or "")
        if hasattr(self._plug_end_model, "set_value"):
            self._plug_end_model.set_value(plug_end or "")

        self._active_path_model.set_value(self._controller.active_cable_path() or "")
        self._refresh_known_cables_label()
        self._refresh_active_tree()
        self._refresh_segment_slider_limit(clamp_value=True)
        self._refresh_subscription_btn()
        self._refresh_visibility_btn()
        self._build_joint_controls()
        self._update_status("Ready to create a cable.", warn=False)

    def _refresh_active_tree(self):
        """Populate the TreeView with known cable paths and sync selection."""
        paths = self._controller.list_cable_paths()
        self._active_tree_model.set_paths(paths)

        # Enable/disable view.
        if self._active_tree_view and hasattr(self._active_tree_view, "enabled"):
            self._active_tree_view.enabled = bool(paths)

        # Clear selection before setting a new one.
        self._syncing_active_selection = True
        try:
            if self._active_tree_view and hasattr(self._active_tree_view, "selection"):
                try:
                    self._active_tree_view.selection = []
                except Exception:
                    pass

            active = self._controller.active_cable_path()
            if paths and active in paths and self._active_tree_view:
                idx = paths.index(active)
                if 0 <= idx < len(self._active_tree_model._items):
                    item = self._active_tree_model._items[idx]
                    if hasattr(self._active_tree_view, "selection"):
                        self._active_tree_view.selection = [item]
        finally:
            self._syncing_active_selection = False

    def _on_active_tree_changed(self, _model=None, _item=None):
        """Handle user selection in the active cable TreeView."""
        if self._syncing_active_selection:
            return

        paths = self._controller.list_cable_paths()
        if not paths or not self._active_tree_view:
            return

        sel = self._active_tree_view.selection if hasattr(self._active_tree_view, "selection") else []
        if not sel:
            return

        item = sel[0]
        if not hasattr(item, "label"):
            return
        path = item.label
        if path not in paths:
            return

        if self._controller.set_active_cable(path):
            self._active_path_model.set_value(path)
            self._refresh_known_cables_label()
            self._build_joint_controls()
            self._update_status(f"Active cable set to {path}.", warn=False)

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

    def _current_joint_limit_span(self, params) -> float:
        try:
            return max(float(getattr(params, "rot_limit_span", 0.0)), 0.0)
        except Exception:
            try:
                return float(getattr(params, "rot_x_high", 0.0) - getattr(params, "rot_x_low", 0.0))
            except Exception:
                return 0.0

    def _joint_limit_hint_text(self, span: float) -> str:
        half = max(span * 0.5, 0.0)
        return f"Lower limit: {-half:.1f}   Upper limit: {half:.1f} (rotX/rotY/rotZ)"

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

        has_cables = bool(self._controller.list_cable_paths())
        subscribed_any = self._controller.any_curve_subscription_active()

        self._subscription_btn.text = (
            "Unsubscribe splines (all)" if subscribed_any else "Subscribe splines (all)"
        )
        self._subscription_btn.enabled = has_cables

    def _refresh_visibility_btn(self):
        if not self._toggle_vis_btn:
            return
        has_cables = bool(self._controller.list_cable_paths())
        show_curve = self._controller.showing_curve_state()
        self._toggle_vis_btn.text = "Show collisions (all)" if show_curve else "Show spline (all)"
        self._toggle_vis_btn.enabled = has_cables

    def _on_sync_all_splines_button(self):
        """Toggle spline updates for all known cables."""
        paths = self._controller.list_cable_paths()
        if not paths:
            self._update_status("No cables to sync. Discover or import first.", warn=True)
            return

        if self._controller.any_curve_subscription_active():
            self._controller.stop_curve_updates_all()
            self._update_status("Stopped spline updates for all cables.", warn=False)
        else:
            self._controller.start_curve_updates_all()
            self._update_status("Spline updates active for all cables.", warn=False)

        self._refresh_subscription_btn()

    def _build_joint_controls(self):
        if not self._joint_frame:
            return

        self._joint_frame.clear()
        self._joint_slider_models = {}
        data = self._controller.get_joint_control_data()

        # Local offsets (as shown in Isaac Sim D6 Joint Properties -> Local Offsets).
        # We use body0 local orientation XYZ (Euler degrees) to seed the UI sliders on discovery/import.
        offsets_by_index = {}
        try:
            offsets = self._controller.get_joint_local_offsets()
            offsets_by_index = {o.get("index"): o for o in offsets}
        except Exception:
            offsets_by_index = {}

        self._syncing_joint_build = True
        try:
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
                                    # Prefer imported local offset orientation for initial slider value.
                                    init_val = targets.get(axis, 0.0)
                                    offs = offsets_by_index.get(idx)
                                    used_offset = False
                                    if offs:
                                        euler = offs.get("local_rot0_euler")
                                        if euler and len(euler) == 3:
                                            if axis == "rotX":
                                                init_val = float(euler[0])
                                                used_offset = True
                                            elif axis == "rotY":
                                                init_val = float(euler[1])
                                                used_offset = True
                                            elif axis == "rotZ":
                                                init_val = float(euler[2])
                                                used_offset = True

                                    # Seed controller targets from local offsets so editing one joint
                                    # doesn't implicitly zero others.
                                    if used_offset:
                                        try:
                                            self._controller.set_joint_drive_target(idx, axis, init_val, apply_pose=True)
                                        except Exception:
                                            pass

                                    model = ui.SimpleFloatModel(init_val)
                                    model.add_value_changed_fn(
                                        lambda m, i=idx, ax=axis: self._on_joint_slider_changed(i, ax, m.as_float)
                                    )
                                    with ui.VStack(height=0, spacing=2):
                                        ui.Label(f"{axis}", width=24, style=get_style())
                                        with ui.HStack(height=18, spacing=4):
                                            ui.FloatSlider(min=low, max=high, model=model, style=get_style(), height=18)
                                            ui.Button(
                                                "·",
                                                width=18,
                                                height=18,
                                                alignment=ui.Alignment.CENTER,
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
                                                        "min_width": 18,
                                                        "max_width": 18,
                                                        "min_height": 18,
                                                        "max_height": 18,
                                                    },
                                                    "Button:hovered": {"background_color": cl(0xFF42A5F5)},
                                                    "Button:pressed": {"background_color": cl(0xFF1E88E5)},
                                                },
                                            )
                                    self._joint_slider_models[(idx, axis)] = model
        finally:
            self._syncing_joint_build = False

    def _clear_joint_controls(self):
        if not self._joint_frame:
            return
        self._joint_frame.clear()
        self._joint_slider_models = {}
        with self._joint_frame:
            ui.Label("Create a cable to edit joint drive targets.", style=get_style())

    def _on_joint_slider_changed(self, joint_index: int, axis: str, value: float):
        if getattr(self, "_syncing_joint_build", False):
            return
        clamped = self._controller.set_joint_drive_target(joint_index, axis, value, apply_pose=True)
        model = self._joint_slider_models.get((joint_index, axis))
        if model and abs(model.as_float - clamped) > 1e-6:
            model.set_value(clamped)
