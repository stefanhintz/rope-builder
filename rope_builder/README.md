# Cable Builder

Cable Builder is an Isaac Sim extension for creating and shaping lightweight **cable** assets made of rigid segments and D6 joints. 
Each cable has two built‑in anchors and optional shape handles so you can quickly pose cables between connection points while keeping collisions and length under control.


## Loading the Extension

Place folder in Documents\Kit\shared\exts\

Once enabled, the extension appears in the toolbar as **Cable Builder** and opens a dockable window on the left of the viewport.


## Core Workflow

The UI is organized into three main sections that match the typical user flow.

### 1. Cables

- Use **Discover cables** to find existing cable roots in `/World`.
- Select a cable in the **Cables** tree to make it active.
- Use **Delete cable** to remove the currently active cable from the stage and controller.

### 2. Shaping (Anchors & Handles)

This section is used most of the time when laying out cables in a scene.

- **Anchors**
  - Each cable has two internal anchors: `anchor_start` and `anchor_end` under the cable root.
  - Place plug meshes at these anchors and parent / rigidly join them to the segment start/end null objects.
  - Move the anchors (not the plug meshes) to the final connection points (sockets, device ports, etc.).

- **Fit cable**
  - Press **Fit cable** to reposition all segments along a smooth curve that passes through the anchors (and any shape handles).
  - Segment lengths are preserved; joint limits can be exceeded if needed to reach the anchors.

- **Shape handles**
  - Press **Add shape handle** to create small handle prims under the cable root.
  - Move these handles in the scene to sculpt the cable path, then press **Fit cable** again to update the pose.

- **Visibility (collisions vs mesh)**
  - Use **Show collisions / Show mesh** to toggle between the collision capsules and the rendered cable mesh to check clearance.

- **Cable length feedback**
  - The **Cable length** row shows:
    - **Original**: the designed cable length from the creation parameters.
    - **Current path**: the length of the current fitted curve between anchors and handles.
  - Comparing these is a quick way to see if the cable is likely overstretched or compressed before running simulation.

During simulation, shape handles are hidden automatically and restored when the timeline is stopped.

### 3. Create new cable

The **Create new cable** section (collapsed by default) is used when you want to author a new cable:

- Configure cable parameters:
  - Total **Length**, **Radius**, **Segments**, and **Total mass**.
  - Joint **limits** (span in degrees) and **drive** settings (stiffness, damping, max force).
- Press **Create cable** to build a new cable root with segments, joints, anchors, curve, and collision geometry.

Once a cable is created, it appears in the **Cables** list and can be shaped via the **Shaping (Anchors & Handles)** section.


## Advanced: Joint Tuning

The **Joint tuning (advanced)** section exposes lower‑level joint controls:

- **Subscribe splines** keeps the internal curve updated from segment poses over time.
- Per‑joint drive target sliders let you adjust D6 joint angles directly for fine‑tuning or debugging.

Most users can ignore this section and work only with anchors, handles, and the **Fit cable** button.


## Code Overview

- `global_variables.py`  
  Defines the extension title and description used by the Cable Builder window.

- `extension.py`  
  Boilerplate entry point that registers the **Cable Builder** window, menu item, and event callbacks, and delegates UI logic to `UIBuilder`.

- `ui_builder.py`  
  Builds the Cable Builder UI (Cables list, Shaping, Create new cable, Joint tuning) and connects widgets to controller methods.

- `scenario.py`  
  Implements `RopeBuilderController`, which creates cables, manages segments, joints, anchors, curve fitting, shape handles, collisions, and length reporting.
