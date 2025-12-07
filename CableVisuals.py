# Cable spline from control points (game approach)
# Paste into Isaac Sim Script Editor / Python Console.

from pxr import Usd, UsdGeom, Gf, Vt
import omni.usd
import omni.kit.app

# ---------------------------
# 1) CONFIG: put your control point prim paths here, in order
# ---------------------------
CONTROL_POINTS = [
    "/World/Cable/plug_negative",
    "/World/Cable/cable/segment_00",
    "/World/Cable/cable/segment_01",
    "/World/Cable/cable/segment_02",
    "/World/Cable/cable/segment_03",
    "/World/Cable/cable/segment_04",
     "/World/Cable/plug_positive",
]

CURVE_PATH = "/World/CableVisual"
CURVE_WIDTH = 0.01   # meters. 0.01 = 1cm diameter-ish look
CURVE_TYPE = "catmullRom"  # smooth spline through points


# ---------------------------
# 2) GET STAGE
# ---------------------------
stage = omni.usd.get_context().get_stage()
if stage is None:
    raise RuntimeError("No USD stage found. Open a stage first.")


# ---------------------------
# 3) CREATE OR GET CURVE PRIM
# ---------------------------
curve_prim = stage.GetPrimAtPath(CURVE_PATH)
if not curve_prim.IsValid():
    curve_prim = UsdGeom.BasisCurves.Define(stage, CURVE_PATH).GetPrim()

curves = UsdGeom.BasisCurves(curve_prim)

# Correct USD tokens:
# type: linear | cubic
# basis: bezier | bspline | catmullRom
# wrap: nonperiodic | periodic | pinned

curves.CreateTypeAttr("cubic")
curves.CreateBasisAttr("catmullRom")
curves.CreateWrapAttr("pinned")

curves.CreateWidthsAttr(Vt.FloatArray([CURVE_WIDTH]))  # constant width
curves.CreateDisplayColorAttr(
    Vt.Vec3fArray([Gf.Vec3f(0.05, 0.05, 0.05)])
)

N = len(CONTROL_POINTS)
curves.CreateCurveVertexCountsAttr(Vt.IntArray([N]))

points_attr = curves.GetPointsAttr()
if not points_attr:
    points_attr = curves.CreatePointsAttr()


# ---------------------------
# 4) HELPER: read world position of a prim
# ---------------------------
def get_world_pos(prim_path: str):
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return None
    xf = UsdGeom.Xformable(prim)
    # ComputeLocalToWorldTransform(timeCode) -> Gf.Matrix4d
    m = xf.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    p = m.ExtractTranslation()
    return Gf.Vec3f(p[0], p[1], p[2])


# ---------------------------
# 5) UPDATE LOOP
# ---------------------------
subscription = None

def on_update(dt):
    pts = []
    for p in CONTROL_POINTS:
        wp = get_world_pos(p)
        if wp is None:
            # If a prim is missing, skip update this frame
            return
        pts.append(wp)

    # Write points to curve
    points_attr.Set(Vt.Vec3fArray(pts))
    curves.GetCurveVertexCountsAttr().Set(Vt.IntArray([len(pts)]))

# Subscribe to per-frame update
app = omni.kit.app.get_app()
subscription = app.get_update_event_stream().create_subscription_to_pop(on_update)

print(f"✅ Cable spline updating from {len(CONTROL_POINTS)} control points.")
print(f"   Curve prim: {CURVE_PATH}")
print("   Stop it later with: subscription.unsubscribe()")