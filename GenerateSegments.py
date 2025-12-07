from pxr import Usd, UsdGeom, UsdPhysics, Gf, Sdf
import omni.usd

# ----------------------------
# Config
# ----------------------------
ROOT_PATH = "/World/cable"
NUM_SEGMENTS =5

SEG_LEN  = 0.1   # 100 mm
SEG_MASS = 0.1   # 100 g

CAP_RADIUS    = 0.01           # 10 mm
CAP_TOTAL_LEN = SEG_LEN * 0.9  # total capsule length incl. hemispheres
CAP_HEIGHT = CAP_TOTAL_LEN - 2.0 * CAP_RADIUS
if CAP_HEIGHT < 0:
    raise ValueError("Capsule height < 0. Increase segment length or reduce radius.")

# ---- Limits (degrees, as UI shows) ----
ROT_X_LOW, ROT_X_HIGH = -30.0, 30.0
ROT_Y_LOW, ROT_Y_HIGH = -30.0, 30.0
ROT_Z_LOW, ROT_Z_HIGH = -30.0, 30.0

# ---- Drives (stiffness + damping) ----
# Start modest, then tune.
DRIVE_STIFFNESS = 1200.0
DRIVE_DAMPING   = 70.0
DRIVE_MAX_FORCE = 200.0   # torque cap; keep reasonable

# ----------------------------
# Get stage
# ----------------------------
stage = omni.usd.get_context().get_stage()
if stage is None:
    raise RuntimeError("No USD stage found. Open a stage first.")

# ----------------------------
# Clean old cable
# ----------------------------
old = stage.GetPrimAtPath(ROOT_PATH)
if old.IsValid():
    stage.RemovePrim(ROOT_PATH)

# ----------------------------
# Helpers
# ----------------------------
def define_xform(path):
    return UsdGeom.Xform.Define(stage, path).GetPrim()

def apply_limit(joint_prim, instance, low, high):
    """
    instance: "rotX"|"rotY"|"rotZ" or "transX"|"transY"|"transZ"
    low/high in degrees for rot*, meters for trans*.
    Shows in UI Limit section.
    """
    lim = UsdPhysics.LimitAPI.Apply(joint_prim, instance)
    lim.CreateLowAttr(low)
    lim.CreateHighAttr(high)

def apply_rot_drive(joint_prim, instance, stiffness, damping, max_force):
    """
    instance: "rotX"|"rotY"|"rotZ"
    Shows in UI Drive section.
    """
    drv = UsdPhysics.DriveAPI.Apply(joint_prim, instance)
    drv.CreateTypeAttr("force")     # torque drive
    drv.CreateStiffnessAttr(stiffness)
    drv.CreateDampingAttr(damping)
    drv.CreateMaxForceAttr(max_force)
    drv.CreateTargetPositionAttr(0.0)
    drv.CreateTargetVelocityAttr(0.0)

# ----------------------------
# Create root
# ----------------------------
define_xform(ROOT_PATH)

# ----------------------------
# Create segments + colliders
# ----------------------------
segment_paths = []

for i in range(NUM_SEGMENTS):
    seg_name = f"segment_{i:02d}"
    seg_path = f"{ROOT_PATH}/{seg_name}"
    segment_paths.append(seg_path)

    seg_prim = define_xform(seg_path)
    seg_xf = UsdGeom.Xformable(seg_prim)
    seg_xf.AddTranslateOp().Set(Gf.Vec3f(i * SEG_LEN, 0.0, 0.0))

    # rigid body + mass
    UsdPhysics.RigidBodyAPI.Apply(seg_prim)
    mass_api = UsdPhysics.MassAPI.Apply(seg_prim)
    mass_api.CreateMassAttr(SEG_MASS)

    # capsule collider
    cap_path = f"{seg_path}/collision"
    cap_prim = UsdGeom.Capsule.Define(stage, cap_path).GetPrim()
    cap = UsdGeom.Capsule(cap_prim)
    cap.CreateAxisAttr("X")
    cap.CreateRadiusAttr(CAP_RADIUS)
    cap.CreateHeightAttr(CAP_HEIGHT)
    UsdPhysics.CollisionAPI.Apply(cap_prim)

# ----------------------------
# Create D6 joints + limits + drives
# ----------------------------
for i in range(NUM_SEGMENTS - 1):
    a_path = segment_paths[i]
    b_path = segment_paths[i + 1]
    joint_path = f"{ROOT_PATH}/joint_{i:02d}_{i+1:02d}"

    joint_prim = UsdPhysics.Joint.Define(stage, joint_path).GetPrim()
    joint = UsdPhysics.Joint(joint_prim)

    joint.CreateBody0Rel().SetTargets([a_path])
    joint.CreateBody1Rel().SetTargets([b_path])

    # joint frames at segment interface
    joint.CreateLocalPos0Attr(Gf.Vec3f( SEG_LEN * 0.5, 0.0, 0.0))
    joint.CreateLocalPos1Attr(Gf.Vec3f(-SEG_LEN * 0.5, 0.0, 0.0))

    # lock translations by "locked" trans limits (low > high)
    for ax in ("transX", "transY", "transZ"):
        apply_limit(joint_prim, ax, 1.0, -1.0)

    # rotation limits (UI Limit panel)
    apply_limit(joint_prim, "rotX", ROT_X_LOW, ROT_X_HIGH)
    apply_limit(joint_prim, "rotY", ROT_Y_LOW, ROT_Y_HIGH)
    apply_limit(joint_prim, "rotZ", ROT_Z_LOW, ROT_Z_HIGH)

    # rotation drives (UI Drive panel)
    apply_rot_drive(joint_prim, "rotX", DRIVE_STIFFNESS, DRIVE_DAMPING, DRIVE_MAX_FORCE)
    apply_rot_drive(joint_prim, "rotY", DRIVE_STIFFNESS, DRIVE_DAMPING, DRIVE_MAX_FORCE)
    apply_rot_drive(joint_prim, "rotZ", DRIVE_STIFFNESS, DRIVE_DAMPING, DRIVE_MAX_FORCE)

print("✅ Cable created with rotX/Y/Z limits and drives.")
print(f"Drive stiffness={DRIVE_STIFFNESS}, damping={DRIVE_DAMPING}, maxForce={DRIVE_MAX_FORCE}")