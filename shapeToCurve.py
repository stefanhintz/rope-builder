from pxr import UsdGeom, Gf
import omni.usd
import math, re

# ----------------------------
# USER INPUT
# ----------------------------
ROOT_PATH = "/World/cable"
RADIUS_M  = 0.5      # arc radius in meters
SEG_LEN_M = 0.1      # <-- REAL segment length (100 mm)
PLANE     = "XZ"     # "XZ" or "XY"

# ----------------------------
# Get stage
# ----------------------------
stage = omni.usd.get_context().get_stage()
if stage is None:
    raise RuntimeError("No USD stage found. Open a stage first.")

root_prim = stage.GetPrimAtPath(ROOT_PATH)
if not root_prim.IsValid():
    raise RuntimeError(f"Root prim not found: {ROOT_PATH}")

# ----------------------------
# Collect segments, sorted by index
# ----------------------------
seg_regex = re.compile(r"segment_(\d+)$")
segments = []
for child in root_prim.GetChildren():
    m = seg_regex.search(child.GetName())
    if m:
        segments.append((int(m.group(1)), child))
segments.sort(key=lambda x: x[0])
seg_prims = [p for _, p in segments]

N = len(seg_prims)
if N < 2:
    raise RuntimeError("Need at least 2 segments.")

# We need N+1 anchors for N segments
num_anchors = N + 1

# IMPORTANT: space anchors by either chord length or arc length?
# To keep centers exactly SEG_LEN apart (like your straight case),
# we space anchors by CHORD ≈ SEG_LEN.
# For small angles chord≈arc anyway, and this keeps touching.
total_angle = (num_anchors - 1) * SEG_LEN_M / RADIUS_M
total_angle = max(0.0, min(total_angle, 2*math.pi*0.95))
dtheta = total_angle / (num_anchors - 1)
start_theta = -total_angle / 2.0

# ----------------------------
# Build anchor points along arc
# ----------------------------
anchors = []
for j in range(num_anchors):
    theta = start_theta + j * dtheta
    if PLANE == "XZ":
        anchors.append(Gf.Vec3f(
            RADIUS_M * math.sin(theta),
            0.0,
            RADIUS_M * (1.0 - math.cos(theta))
        ))
    elif PLANE == "XY":
        anchors.append(Gf.Vec3f(
            RADIUS_M * math.sin(theta),
            RADIUS_M * (1.0 - math.cos(theta)),
            0.0
        ))
    else:
        raise ValueError("PLANE must be 'XZ' or 'XY'")

# ----------------------------
# Place segments between anchors, align local X to chord
# ----------------------------
local_x = Gf.Vec3d(1.0, 0.0, 0.0)

for i, prim in enumerate(seg_prims):
    p0 = Gf.Vec3d(anchors[i])
    p1 = Gf.Vec3d(anchors[i + 1])

    center = (p0 + p1) * 0.5
    t = (p1 - p0)
    if t.GetLength() < 1e-9:
        continue
    t_norm = t.GetNormalized()

    # rotate local +X onto tangent/chord
    rot = Gf.Rotation(local_x, t_norm)
    q = rot.GetQuat()  # GfQuatd
    qf = Gf.Quatf(float(q.GetReal()), Gf.Vec3f(q.GetImaginary()))  # cast to float

    xf = UsdGeom.Xformable(prim)
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3f(center))
    xf.AddOrientOp().Set(qf)

print(f"✅ Arranged {N} segments on arc.")
print(f"   radius={RADIUS_M}m, segment length={SEG_LEN_M}m")
print(f"   total angle ≈ {math.degrees(total_angle):.1f}°")