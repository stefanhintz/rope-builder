# SPDX-License-Identifier: Apache-2.0

"""Script Node: update the visual curve for an existing Cable Builder cable.

Suggested inputs:
    cablePath: string/token, required. Example: "/World/cable"
    enabled: bool, default True
    timelinePlaying: bool, default True
    updateWhileStopped: bool, default False
    frame: int, default 0
    frameStride: int, default 1
    curveExtension: float, default 0.02
    curveWidthScale: float, default 2.0

Suggested outputs:
    didUpdate: bool
    message: string

This script intentionally does not import the Cable Builder extension.
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

import carb
import omni.usd
from pxr import Gf, Sdf, Usd, UsdGeom, Vt


def compute(db):
    root_path = _input(db, "cablePath", "")
    enabled = bool(_input(db, "enabled", True))
    timeline_playing = bool(_input(db, "timelinePlaying", True))
    update_while_stopped = bool(_input(db, "updateWhileStopped", False))
    frame = int(_input(db, "frame", 0) or 0)
    frame_stride = max(int(_input(db, "frameStride", 1) or 1), 1)
    extension = float(_input(db, "curveExtension", 0.02) or 0.0)
    width_scale = float(_input(db, "curveWidthScale", 2.0) or 2.0)

    try:
        if not enabled:
            return _finish(db, False, "Curve update disabled.")
        if not timeline_playing and not update_while_stopped:
            return _finish(db, False, "Timeline stopped; curve update skipped.")
        if frame % frame_stride != 0:
            return _finish(db, False, f"Frame {frame} skipped by frameStride {frame_stride}.")
        if not root_path:
            raise RuntimeError("Input cablePath is empty.")

        stage = omni.usd.get_context().get_stage()
        if not stage:
            raise RuntimeError("No open USD stage.")

        segment_paths = _segment_paths(stage, root_path)
        if len(segment_paths) < 2:
            raise RuntimeError(f"Need at least two cable segments under {root_path}.")

        update_curve(stage, root_path, segment_paths, extension, width_scale)
        return _finish(db, True, f"Updated curve for {root_path}.")
    except Exception as exc:
        message = str(exc)
        carb.log_warn(f"[CableCurveNode] {message}")
        _output(db, "didUpdate", False)
        _output(db, "message", message)
        return False


def update_curve(stage, root_path: str, segment_paths: List[str], extension: float, width_scale: float):
    curve_path = f"{root_path}/curve"
    curves = UsdGeom.BasisCurves.Get(stage, Sdf.Path(curve_path))
    if not curves:
        curves = UsdGeom.BasisCurves.Define(stage, Sdf.Path(curve_path))
    curve_prim = curves.GetPrim()
    curves.CreateTypeAttr(UsdGeom.Tokens.cubic).Set(UsdGeom.Tokens.cubic)
    curves.CreateBasisAttr(UsdGeom.Tokens.bspline).Set(UsdGeom.Tokens.bspline)
    curves.CreateWrapAttr(UsdGeom.Tokens.pinned).Set(UsdGeom.Tokens.pinned)

    points = []
    first_pose = _world_frame(stage, segment_paths[0])
    last_pose = _world_frame(stage, segment_paths[-1])
    seg_lengths = _segment_lengths(stage, segment_paths)
    extension = max(float(extension), 0.0)

    if first_pose:
        pos, rot = first_pose
        dir_x = Gf.Rotation(rot).TransformDir(Gf.Vec3d(1.0, 0.0, 0.0))
        tip = _world_pos(stage, f"{segment_paths[0]}/tip")
        if tip is None:
            tip = pos - dir_x * (float(seg_lengths[0]) * 0.5)
        points.append(tip - dir_x * extension)

    for path in segment_paths:
        pos = _world_pos(stage, f"{path}/collision") or _world_pos(stage, path)
        if pos is not None:
            points.append(pos)

    if last_pose:
        pos, rot = last_pose
        dir_x = Gf.Rotation(rot).TransformDir(Gf.Vec3d(1.0, 0.0, 0.0))
        tip = _world_pos(stage, f"{segment_paths[-1]}/tip")
        if tip is None:
            tip = pos + dir_x * (float(seg_lengths[-1]) * 0.5)
        points.append(tip + dir_x * extension)

    counts_attr = curves.GetCurveVertexCountsAttr() or curves.CreateCurveVertexCountsAttr()
    points_attr = curves.GetPointsAttr() or curves.CreatePointsAttr()
    if len(points) < 2:
        counts_attr.Set(Vt.IntArray([0]))
        points_attr.Set(Vt.Vec3fArray())
        return

    inv = UsdGeom.Xformable(curve_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()).GetInverse()
    local_points = [Gf.Vec3f(inv.Transform(point)) for point in points]
    counts_attr.Set(Vt.IntArray([len(local_points)]))
    points_attr.Set(Vt.Vec3fArray(local_points))

    radius_attr = stage.GetPrimAtPath(f"{segment_paths[0]}/collision").GetAttribute("radius")
    radius = float(radius_attr.Get()) if radius_attr and radius_attr.HasAuthoredValueOpinion() else 0.01
    curves.CreateWidthsAttr(Vt.FloatArray([max(radius * max(width_scale, 0.0), 1e-4)]))


def _segment_paths(stage, root_path: str) -> List[str]:
    root = stage.GetPrimAtPath(root_path)
    if not root or not root.IsValid():
        raise RuntimeError(f"Root prim not found: {root_path}")

    numbered = []
    regex = re.compile(r"segment_(\d+)$")
    for child in root.GetChildren():
        match = regex.search(child.GetName())
        if match:
            numbered.append((int(match.group(1)), child.GetPath().pathString))
    numbered.sort(key=lambda item: item[0])

    paths = [path for _, path in numbered]
    start = f"{root_path}/segment_start"
    end = f"{root_path}/segment_end"
    if stage.GetPrimAtPath(start).IsValid():
        paths.insert(0, start)
    if stage.GetPrimAtPath(end).IsValid():
        paths.append(end)
    return paths


def _segment_lengths(stage, segment_paths: List[str]) -> List[float]:
    lengths = []
    for idx, path in enumerate(segment_paths):
        length = None
        radius = 0.0
        col = stage.GetPrimAtPath(f"{path}/collision")
        if col and col.IsValid():
            radius_attr = col.GetAttribute("radius")
            height_attr = col.GetAttribute("height")
            if radius_attr and radius_attr.HasAuthoredValueOpinion():
                radius = float(radius_attr.Get())
            if height_attr and height_attr.HasAuthoredValueOpinion():
                length = float(height_attr.Get()) + 2.0 * radius
        if length is None and idx + 1 < len(segment_paths):
            p0 = _world_pos(stage, path)
            p1 = _world_pos(stage, segment_paths[idx + 1])
            if p0 is not None and p1 is not None:
                length = float((p1 - p0).GetLength())
        lengths.append(length if length is not None else 0.1)
    return lengths


def _world_frame(stage, path) -> Optional[Tuple[Gf.Vec3d, Gf.Quatd]]:
    prim = stage.GetPrimAtPath(path)
    if not prim or not prim.IsValid():
        return None
    matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    return Gf.Vec3d(matrix.ExtractTranslation()), matrix.ExtractRotation().GetQuat()


def _world_pos(stage, path) -> Optional[Gf.Vec3d]:
    frame = _world_frame(stage, path)
    return frame[0] if frame else None


def _finish(db, did_update: bool, message: str):
    if did_update:
        carb.log_info(f"[CableCurveNode] {message}")
    _output(db, "didUpdate", did_update)
    _output(db, "message", message)
    return did_update


def _input(db, name, default=None):
    inputs = getattr(db, "inputs", None)
    return getattr(inputs, name, default) if inputs is not None else default


def _output(db, name, value):
    outputs = getattr(db, "outputs", None)
    if outputs is not None and hasattr(outputs, name):
        setattr(outputs, name, value)


if "db" in globals():
    compute(db)
