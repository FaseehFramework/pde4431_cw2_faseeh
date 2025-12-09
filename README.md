# PRRR Robot Arm Simulation

An interactive 3D simulation of a PRRR (Prismatic-Revolute-Revolute-Revolute) 4-DOF robot manipulator with pick-and-place capabilities and smart stacking logic.

# IMPORTANT

Please do not move mouse while the simulation is running over the command center window. It may cause unexpected behavior. Or you may drag the cursor outside the command center window.

## Overview
This script simulates a PRRR robot arm that can manipulate objects in a 3D workspace. The robot features:

- **PRRR Configuration**: 1 prismatic joint (vertical lift) + 3 revolute joints (planar arm)
- **Real-time 3D Visualization**: Interactive matplotlib-based rendering
- **Pick-and-Place Operations**: Animated object manipulation
- **Smart Stacking**: Automatic height calculation for stacking objects

## Robot Configuration

### D-H Parameters

| Joint | Type      | θ (theta) | d       | a    | α (alpha) |
|-------|-----------|-----------|---------|------|-----------|
| 1     | Prismatic | 0°        | **d₁**  | 0    | 0°        |
| 2     | Revolute  | **θ₁**    | 0       | 3.5  | 0°        |
| 3     | Revolute  | **θ₂**    | 0       | 2.5  | 0°        |
| 4     | Revolute  | **θ₃**    | 0       | 1.5  | 0°        |

### Workspace Limits

- **Max Z Height**: 5.0 units
- **Max Reach**: 7.5 units (3.5 + 2.5 + 1.5)
- **X/Y Range**: ±8 units (visualization bounds)

## Installation

### Dependencies

```bash
pip install numpy matplotlib
```

### Running the Simulation

```bash
python PRRR_ani.py
```

## User Interface

The simulation opens two windows:

### 1. Visualization Window
- 3D view of the robot arm and objects
- Robot links shown in different colors:
  - **Black**: Prismatic joint (vertical)
  - **Blue**: Shoulder link
  - **Red**: Elbow link
  - **Green**: Wrist link
- Objects displayed with their respective colors and shapes
- Ghost markers (faded) show destination positions

### 2. Command Center (Control Panel)

| Section | Control | Description |
|---------|---------|-------------|
| **Visuals** | Show Workspace | Toggle cylindrical workspace envelope |
| **Individual Tasks** | Move Triangle | Move green triangle to its destination |
| | Move Circle | Move yellow circle to its destination |
| **Blue Square Control** | Dest (x y z) + GO | Move blue square to specified user inputted coordinates |
| **Master Stack Control** | Target (x y z) + STACK ALL | Stack all objects at specified location |
| **Reset** | RESET SIMULATION | Reset all objects and robot to initial state |

## Objects

| Object | Color | Marker | Initial Position | Destination |
|--------|-------|--------|------------------|-------------|
| Blue Square | Blue | ■ | Random (3-6, -3 to 3, 0) | User-defined |
| Green Triangle | Green | ▲ | Random (3-6, -3 to 3, 0) | Random(x,y) Z=2.0 |
| Yellow Circle | Yellow | ● | Random (3-6, -3 to 3, 0) | Random(x,y) Z=4.0 |

> **Note**: Destination coordinates for the Green Triangle and Yellow Circle are printed to the terminal on startup.

## Features

### Smart Stacking Logic

The robot automatically calculates the correct Z-height when placing objects:

1. **Collision Detection**: Checks if another object exists at target (X, Y)
2. **Height Calculation**: `target_z = max(input_z, existing_stack_height)`
3. **Skip Optimization**: Objects already at target location are skipped

### Pick-and-Place Sequence

1. Move to hover position above object
2. Descend to object position
3. "Grab" object (turns red)
4. Lift to safe travel height
5. Move horizontally to destination
6. Descend to target height
7. "Release" object (returns to original color)
8. Retract to hover position

### Kinematics

- **Forward Kinematics**: Computes joint positions from joint variables
- **Inverse Kinematics**: Calculates joint variables from target position
  - Uses geometric approach with wrist position decoupling
  - Handles singularities and unreachable positions

## Configuration Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `STEP_SIZE_ANG` | 5.0° | Angular step size for animation |
| `STEP_SIZE_LIN` | 0.2 | Linear step size for prismatic joint |
| `SAFE_HOVER_HEIGHT` | 2.0 | Height offset during travel |
| `OBJECT_HEIGHT` | 0.6 | Height of each block for stacking |
| `STACK_TOLERANCE` | 0.5 | Distance threshold for stacking detection |

## Terminal Output

The script prints useful information to the terminal:

- Object destination coordinates on startup
- Master Stack execution details
- Object movement status (moving/skipping)
- Reset notifications

## Example Usage

1. **Run the script**: `python PRRR_ani.py`
2. **Note the printed destinations** for triangle and circle
3. **Click "Move Triangle"** to move it to its destination
4. **Enter coordinates** in the Blue Square Control (e.g., `2 3 1`) and click GO
5. **Use Master Stack** to gather all objects at said destination

## File Structure

```
PRRR_ani.py
├── Configuration & Parameters
│   ├── DH_TABLE
│   ├── Robot Geometry Constants
│   └── Simulation Settings
├── Classes
│   ├── WorldObject
│   └── RobotController
│       ├── __init__()
│       ├── init_world()
│       ├── setup_control_panel()
│       ├── get_smart_z()
│       ├── on_stack_all_submit()
│       ├── run_task()
│       ├── on_square_submit()
│       ├── toggle_workspace()
│       ├── reset_sim()
│       ├── generate_workspace_data()
│       ├── forward_kinematics()
│       ├── inverse_kinematics()
│       ├── move_to()
│       ├── pick_and_place()
│       └── draw_scene()
└── Main Execution
```

## License

Educational use for PDE4431 coursework.
