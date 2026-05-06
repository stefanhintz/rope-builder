# SPDX-License-Identifier: Apache-2.0

"""Script Node: fit an existing Cable Builder cable to anchors and handles.

Suggested inputs:
    cablePath: string/token, optional. Example: "/World/cable".
        If empty, the script uses the parent cable prim of the containing Action Graph.
    keepExactLength: bool, default True

Suggested outputs:
    success: bool
    ropeLength: float
    pathLength: float
    straightPossible: bool
    hasSlack: bool
    slackLength: float
    stretchLength: float
    message: string

This script intentionally does not import the Cable Builder extension.
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

import carb
import omni.usd
from pxr import Gf, Sdf, Usd, UsdGeom, Vt


CURVE_EXTENSION = 0.02
CURVE_WIDTH_SCALE = 2.0
LENGTH_EPS = 1e-4


def compute(db):
    keep_exact_length = bool(_input(db, "keepExactLength", True))
    try:
        stage = omni.usd.get_context().get_stage()
        if not stage:
            raise RuntimeError("No open USD stage.")

        root_path = _resolve_root_path(db, stage)

        segment_paths = _segment_paths(stage, root_path)
        if len(segment_paths) < 2:
            raise RuntimeError(f"Need at least two cable segments under {root_path}.")

        rope_len, path_len, straight_possible, has_slack, slack_len, stretch_len = fit_cable_to_anchors(
            stage, root_path, segment_paths, keep_exact_length
        )
        update_curve(stage, root_path, segment_paths)
        if straight_possible:
            fit_msg = "straight possible"
        elif has_slack:
            fit_msg = f"slack {slack_len:.3f} m"
        else:
            fit_msg = f"overstretched by {stretch_len:.3f} m"
        message = f"Fitted {root_path}: cable {rope_len:.3f} m, path {path_len:.3f} m ({fit_msg})."
        carb.log_info(f"[CableFitNode] {message}")
        _output(db, "success", True)
        _output(db, "ropeLength", rope_len)
        _output(db, "pathLength", path_len)
        _output(db, "straightPossible", straight_possible)
        _output(db, "hasSlack", has_slack)
        _output(db, "slackLength", slack_len)
        _output(db, "stretchLength", stretch_len)
        _output(db, "message", message)
        return True
    except Exception as exc:
        message = str(exc)
        carb.log_warn(f"[CableFitNode] {message}")
        _output(db, "success", False)
        _output(db, "message", message)
        return False


def fit_cable_to_anchors(
    stage, root_path: str, segment_paths: List[str], keep_exact_length: bool
) -> Tuple[float, float, bool, bool, float, float]:
    start_pose = _world_frame(stage, f"{root_path}/anchor_start")
    end_pose = _world_frame(stage, f"{root_path}/anchor_end")
    if not start_pose or not end_pose:
        raise RuntimeError("Missing anchor_start or anchor_end.")

    p0, r0 = start_pose
    p1, r1 = end_pose
    seg_lengths = _segment_lengths(stage, segment_paths)
    rope_len = float(sum(seg_lengths))
    if rope_len <= 1e-6:
        raise RuntimeError("Cable length is zero.")

    dir0 = _safe_dir(Gf.Rotation(r0).TransformDir(Gf.Vec3d(1.0, 0.0, 0.0)))
    dir1 = _safe_dir(Gf.Rotation(r1).TransformDir(Gf.Vec3d(1.0, 0.0, 0.0)))
    start_len = float(seg_lengths[0])
    end_len = float(seg_lengths[-1])
    start_seg = segment_paths[0]
    end_seg = segment_paths[-1]

    start_col = _child_local_offset(stage, start_seg, f"{start_seg}/collision") or Gf.Vec3d(0.0)
    end_col = _child_local_offset(stage, end_seg, f"{end_seg}/collision") or Gf.Vec3d(0.0)
    start_attach = (
        _child_local_offset(stage, start_seg, f"{start_seg}/tip/attach")
        or _child_local_offset(stage, start_seg, f"{start_seg}/tip")
        or start_col + Gf.Vec3d(-0.5 * start_len, 0.0, 0.0)
    )
    end_attach = (
        _child_local_offset(stage, end_seg, f"{end_seg}/tip/attach")
        or _child_local_offset(stage, end_seg, f"{end_seg}/tip")
        or end_col + Gf.Vec3d(0.5 * end_len, 0.0, 0.0)
    )

    start_inner = start_col + Gf.Vec3d(0.5 * start_len, 0.0, 0.0)
    end_inner = end_col + Gf.Vec3d(-0.5 * end_len, 0.0, 0.0)
    inner_p0 = p0 + dir0 * float(start_inner[0] - start_attach[0])
    inner_p1 = p1 - dir1 * float(end_attach[0] - end_inner[0])

    delta = inner_p1 - inner_p0
    straight_dist = float(delta.GetLength())
    mid_rope_len = float(rope_len - start_len - end_len)
    straight_path_len = float(start_len + straight_dist + end_len)
    if keep_exact_length and mid_rope_len + LENGTH_EPS < straight_dist:
        raise RuntimeError(
            f"Cannot keep exact length: anchors need {straight_dist + start_len + end_len:.3f} m, "
            f"but cable length is {rope_len:.3f} m."
        )

    ctrl_pts = [inner_p0] + _handle_points(stage, root_path) + [inner_p1]
    handle_count = max(len(ctrl_pts) - 2, 0)
    rope_delta = rope_len - straight_path_len
    slack_len = max(rope_delta, 0.0)
    stretch_len = max(-rope_delta, 0.0)
    has_slack = slack_len > LENGTH_EPS
    straight_possible = handle_count == 0 and abs(rope_delta) <= LENGTH_EPS
    tangent_scale = 0.0 if straight_possible else max(straight_dist * 0.5, rope_len * 0.25, 1e-3)

    if keep_exact_length:
        ctrl_pts, tangent_scale = _adjust_curve_to_length(
            ctrl_pts, handle_count, inner_p0, inner_p1, dir0, dir1, mid_rope_len, straight_dist, tangent_scale
        )

    sampler = _make_catmull_sampler(ctrl_pts, dir0, dir1, tangent_scale)
    curve_len, ts, cumulative = _sample_curve(sampler, max(32, len(segment_paths) * 4))
    use_straight = curve_len <= 1e-6
    if use_straight:
        curve_len = straight_dist

    def sample_pos_and_dir(s: float):
        s = max(0.0, min(1.0, float(s)))
        if use_straight or curve_len <= 1e-6:
            return inner_p0 + delta * s, _safe_dir(delta)
        target_len = s * curve_len
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
            alpha = 0.0 if l1 <= l0 else (target_len - l0) / (l1 - l0)
            t_val = ts[idx - 1] * (1.0 - alpha) + ts[idx] * alpha
        return sampler(t_val)

    start_attach_local = _child_local_offset(stage, start_seg, f"{start_seg}/tip/attach")
    end_attach_local = _child_local_offset(stage, end_seg, f"{end_seg}/tip/attach")
    start_tip_local = _child_local_offset(stage, start_seg, f"{start_seg}/tip")
    end_tip_local = _child_local_offset(stage, end_seg, f"{end_seg}/tip")
    cursor_mid = 0.0
    last_index = len(segment_paths) - 1

    for idx, seg_path in enumerate(segment_paths):
        seg_len = float(seg_lengths[idx])
        if idx == 0:
            world_q = r0
            tip_local = start_attach_local or start_tip_local or Gf.Vec3d(-0.5 * seg_len, 0.0, 0.0)
            origin_world = p0 - Gf.Rotation(world_q).TransformDir(tip_local)
        elif idx == last_index:
            world_q = r1
            tip_local = end_attach_local or end_tip_local or Gf.Vec3d(0.5 * seg_len, 0.0, 0.0)
            origin_world = p1 - Gf.Rotation(world_q).TransformDir(tip_local)
        else:
            mid_s = 0.5 if mid_rope_len <= 1e-6 else (cursor_mid + 0.5 * seg_len) / mid_rope_len
            center_world, tangent_world = sample_pos_and_dir(mid_s)
            q = Gf.Rotation(Gf.Vec3d(1.0, 0.0, 0.0), tangent_world).GetQuat()
            world_q = Gf.Quatd(q.GetReal(), q.GetImaginary())
            col_local = _child_local_offset(stage, seg_path, f"{seg_path}/collision") or Gf.Vec3d(0.0)
            origin_world = center_world - Gf.Rotation(world_q).TransformDir(col_local)
            cursor_mid += seg_len
        _set_local_from_world(stage, seg_path, origin_world, world_q)

    return rope_len, float(start_len + curve_len + end_len), straight_possible, has_slack, slack_len, stretch_len


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


def _resolve_root_path(db, stage) -> str:
    root_path = str(_input(db, "cablePath", "") or "")
    if root_path:
        return root_path

    for candidate in _db_path_candidates(db):
        root_path = _find_cable_ancestor(stage, candidate)
        if root_path:
            return root_path

    raise RuntimeError("Input cablePath is empty and the containing Action Graph is not under a cable root.")


def _db_path_candidates(db) -> List[str]:
    candidates = []
    seen = set()
    objects = [db]
    for attr_name in ("abi_node", "node", "_node", "graph", "_graph"):
        obj = _maybe_call(getattr(db, attr_name, None))
        if obj is not None:
            objects.append(obj)

    for obj in objects:
        for method_name in ("get_prim_path", "get_path", "get_absolute_path", "get_graph_path", "get_path_to_graph"):
            method = getattr(obj, method_name, None)
            if not callable(method):
                continue
            try:
                path = _path_string(method())
            except Exception:
                path = ""
            if path and path not in seen:
                candidates.append(path)
                seen.add(path)

        for attr_name in ("prim_path", "path"):
            path = _path_string(getattr(obj, attr_name, None))
            if path and path not in seen:
                candidates.append(path)
                seen.add(path)

        graph = _maybe_call(getattr(obj, "get_graph", None))
        if graph is not None:
            for method_name in ("get_prim_path", "get_path", "get_absolute_path", "get_graph_path", "get_path_to_graph"):
                method = getattr(graph, method_name, None)
                if not callable(method):
                    continue
                try:
                    path = _path_string(method())
                except Exception:
                    path = ""
                if path and path not in seen:
                    candidates.append(path)
                    seen.add(path)

    return candidates


def _maybe_call(value):
    if callable(value):
        try:
            return value()
        except Exception:
            return value
    return value


def _path_string(value) -> str:
    if value is None:
        return ""
    path = getattr(value, "pathString", None)
    if path:
        return str(path)
    path = str(value)
    return path if path.startswith("/") else ""


def _find_cable_ancestor(stage, path_string: str) -> str:
    try:
        path = Sdf.Path(path_string)
        if not path.IsPrimPath():
            path = path.GetPrimPath()
    except Exception:
        return ""

    while path.pathString not in ("", "/"):
        prim = stage.GetPrimAtPath(path)
        if prim and prim.IsValid() and _is_cable_root_prim(prim):
            return path.pathString
        path = path.GetParentPath()
    return ""


def _is_cable_root_prim(prim) -> bool:
    regex = re.compile(r"segment_(\d+)$")
    for child in prim.GetChildren():
        name = child.GetName()
        if name in ("segment_start", "segment_end") or regex.search(name):
            return True
    return False


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


def _handle_points(stage, root_path: str) -> List[Gf.Vec3d]:
    root = stage.GetPrimAtPath(root_path)
    handles = []
    regex = re.compile(r"handle_(\d+)$")
    for child in root.GetChildren():
        match = regex.search(child.GetName())
        if match:
            pos = _world_pos(stage, child.GetPath().pathString)
            if pos is not None:
                handles.append((int(match.group(1)), pos))
    handles.sort(key=lambda item: item[0])
    return [pos for _, pos in handles]


def _adjust_curve_to_length(points, handle_count, inner_p0, inner_p1, dir0, dir1, target_len, straight_dist, scale):
    if handle_count == 0:
        if target_len <= straight_dist + LENGTH_EPS:
            return [inner_p0, inner_p1], 0.0
        chord_dir = _safe_dir(inner_p1 - inner_p0)
        normal = _cross(chord_dir, Gf.Vec3d(0.0, 0.0, 1.0))
        if normal.GetLength() < 1e-6:
            normal = _cross(chord_dir, Gf.Vec3d(0.0, 1.0, 0.0))
        normal.Normalize()
        midpoint = (inner_p0 + inner_p1) * 0.5

        def sag_points(height):
            return [inner_p0, midpoint + normal * height, inner_p1]

        low = 0.0
        high = max(target_len, straight_dist, 1e-3) * 0.5
        while _curve_length(sag_points(high), dir0, dir1, 0.0) < target_len:
            high *= 2.0
        for _ in range(32):
            mid = (low + high) * 0.5
            if _curve_length(sag_points(mid), dir0, dir1, 0.0) < target_len:
                low = mid
            else:
                high = mid
        return sag_points(high), 0.0

    min_len = _curve_length(points, dir0, dir1, 0.0)
    if min_len > target_len + LENGTH_EPS:
        raise RuntimeError("Cannot keep exact length: handle path is longer than the cable.")

    low = 0.0
    high = max(scale, 1e-3)
    while _curve_length(points, dir0, dir1, high) < target_len and high < max(target_len, 1.0) * 16.0:
        high *= 2.0
    if _curve_length(points, dir0, dir1, high) < target_len - LENGTH_EPS:
        raise RuntimeError("Cannot keep exact length with the current handles and anchor directions.")

    for _ in range(32):
        mid = (low + high) * 0.5
        if _curve_length(points, dir0, dir1, mid) < target_len:
            low = mid
        else:
            high = mid
    return points, high


def _curve_length(points, dir0, dir1, scale):
    return _sample_curve(_make_catmull_sampler(points, dir0, dir1, scale), max(32, len(points) * 8))[0]


def _make_catmull_sampler(points, dir0, dir1, scale):
    ext = [Gf.Vec3d(0.0)] * (len(points) + 2)
    for idx, point in enumerate(points):
        ext[idx + 1] = point
    ext[0] = points[0] - dir0 * scale
    ext[-1] = points[-1] + dir1 * scale
    num_segs = max(len(points) - 1, 1)

    def sample(u):
        u = max(0.0, min(1.0, float(u)))
        s = u * float(num_segs)
        seg = min(int(s), num_segs - 1)
        t = 1.0 if int(s) >= num_segs else s - float(seg)
        p0 = ext[seg]
        p1 = ext[seg + 1]
        p2 = ext[seg + 2]
        p3 = ext[seg + 3]
        t2 = t * t
        t3 = t2 * t
        pos = 0.5 * (
            (2.0 * p1)
            + (-p0 + p2) * t
            + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
            + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3
        )
        tangent = 0.5 * (
            (-p0 + p2)
            + (2.0 * (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3)) * t
            + (3.0 * (-p0 + 3.0 * p1 - 3.0 * p2 + p3)) * t2
        )
        return pos, _safe_dir(tangent)

    return sample


def _sample_curve(sampler, sample_count):
    ts = []
    cumulative = []
    last_pos = None
    length = 0.0
    for idx in range(sample_count):
        t = float(idx) / float(max(sample_count - 1, 1))
        pos, _ = sampler(t)
        ts.append(t)
        if last_pos is not None:
            length += float((pos - last_pos).GetLength())
        cumulative.append(length)
        last_pos = pos
    return length, ts, cumulative


def update_curve(stage, root_path: str, segment_paths: List[str]):
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
    extension = max(CURVE_EXTENSION, 0.0)

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
    curves.CreateWidthsAttr(Vt.FloatArray([max(radius * max(CURVE_WIDTH_SCALE, 0.0), 1e-4)]))


def _set_local_from_world(stage, path, pos, rot):
    prim = stage.GetPrimAtPath(path)
    if not prim or not prim.IsValid():
        return
    parent = prim.GetParent()
    parent_world = Gf.Matrix4d(1.0)
    if parent and parent.IsValid():
        parent_world = UsdGeom.Xformable(parent).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    local_pos = parent_world.GetInverse().Transform(pos)
    local_rot = parent_world.ExtractRotation().GetQuat().GetInverse() * rot
    xf = UsdGeom.Xformable(prim)
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3f(local_pos))
    xf.AddOrientOp().Set(Gf.Quatf(float(local_rot.GetReal()), Gf.Vec3f(local_rot.GetImaginary())))
    xf.AddScaleOp().Set(Gf.Vec3f(1.0, 1.0, 1.0))


def _world_frame(stage, path) -> Optional[Tuple[Gf.Vec3d, Gf.Quatd]]:
    prim = stage.GetPrimAtPath(path)
    if not prim or not prim.IsValid():
        return None
    matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    return Gf.Vec3d(matrix.ExtractTranslation()), matrix.ExtractRotation().GetQuat()


def _world_pos(stage, path) -> Optional[Gf.Vec3d]:
    frame = _world_frame(stage, path)
    return frame[0] if frame else None


def _child_local_offset(stage, parent_path, child_path) -> Optional[Gf.Vec3d]:
    parent = stage.GetPrimAtPath(parent_path)
    child = stage.GetPrimAtPath(child_path)
    if not parent or not parent.IsValid() or not child or not child.IsValid():
        return None
    parent_world = UsdGeom.Xformable(parent).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    child_world = UsdGeom.Xformable(child).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    return Gf.Vec3d(parent_world.GetInverse().Transform(Gf.Vec3d(child_world.ExtractTranslation())))


def _safe_dir(vec):
    out = Gf.Vec3d(vec)
    if out.GetLength() < 1e-6:
        return Gf.Vec3d(1.0, 0.0, 0.0)
    out.Normalize()
    return out


def _cross(a, b):
    return Gf.Vec3d(
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _input(db, name, default=None):
    inputs = getattr(db, "inputs", None)
    return getattr(inputs, name, default) if inputs is not None else default


def _output(db, name, value):
    outputs = getattr(db, "outputs", None)
    if outputs is not None and hasattr(outputs, name):
        setattr(outputs, name, value)


if "db" in globals():
    compute(db)
