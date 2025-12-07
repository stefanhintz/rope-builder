# Usage

To enable this extension, run Isaac Sim with the flags `--ext-folder {path_to_ext_folder} --enable {ext_directory_name}`.

## Cable workflow

- Open the Rope Builder window, set cable length, radius, segment count, and mass.
- Tune joint limits (degrees) plus drive stiffness/damping/force before creating the cable.
- Click **Create Cable** to author lightweight rigid-body capsules under `/World/cable` and D6 joints between them.
- Use **Subscribe spline update** to draw a curve at `/World/cable/curve` that follows segment positions every frame; toggle again to stop.
- Shape the cable interactively with the per-joint rotX/rotY/rotZ sliders; **Reset joint targets** snaps all drives back to zero inside the allowed limits.
- Toggle **Show collisions/Show spline** to swap visibility between the spline visualization and the collision capsules.
- Create multiple cables by giving each a name before clicking **Create Cable**; use **Set active** to pick which cable the sliders and actions control. Import existing cables on stage by entering their root prim path and clicking **Import cable**.

