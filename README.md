# PRRR Robot Arm Simulation

An interactive 3D simulation of a PRRR (Prismatic-Revolute-Revolute-Revolute) 4-DOF robot manipulator with pick-and-place capabilities and smart stacking logic.

# Video Demo [link](https://www.youtube.com/watch?v=pQ4eXLlYepU)

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

![alt text](https://github.com/FaseehFramework/pde4431_cw2_faseeh/blob/master/images/viz%20window.png?raw=true "Visualization Window")

### 2. Command Center (Control Panel)

| Section | Control | Description |
|---------|---------|-------------|
| **Visuals** | Show Workspace | Toggle cylindrical workspace envelope |
| **Individual Tasks** | Move Triangle | Move green triangle to its destination |
| | Move Circle | Move yellow circle to its destination |
| **Blue Square Control** | Dest (x y z) + GO | Move blue square to specified user inputted coordinates |
| **Master Stack Control** | Target (x y z) + STACK ALL | Stack all objects at specified location |
| **Reset** | RESET SIMULATION | Reset all objects and robot to initial state |

![alt text](https://github.com/FaseehFramework/pde4431_cw2_faseeh/blob/master/images/cmd%20center.png?raw=true "Command Center")

## Objects

| Object | Color | Marker | Initial Position | Destination |
|--------|-------|--------|------------------|-------------|
| Blue Square | Blue | ■ | Random (0-6, -3 to 3, 0) | User-defined |
| Green Triangle | Green | ▲ | Random (0-6, -3 to 3, 0) | Random(x,y) Z=2.0 |
| Yellow Circle | Yellow | ● | Random (0-6, -3 to 3, 0) | Random(x,y) Z=4.0 |

> **Note**: Destination coordinates for the Green Triangle and Yellow Circle are printed to the terminal on startup.

## Features

### Smart Stacking Logic

The robot automatically calculates the correct Z-height when placing objects:

1. **Collision Detection**: Checks if another object exists at target (X, Y)
2. **Height Calculation**: `target_z = max(input_z, existing_stack_height)`
3. **Skip Optimization**: Objects already at target location are skipped

### Early Validation

Before attempting to move any object, the robot performs **workspace validation**:

1. **Reachability Check**: Tests if the destination is within the robot's workspace
2. **IK Solution Test**: Attempts to solve inverse kinematics for the target position
3. **Abort on Failure**: If destination is unreachable, prints `"Outside workspace , cannot reach"` and cancels the operation

This prevents the robot from lifting an object only to discover it cannot reach the destination.

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

## Test Cases

### Test Case 1: Move Triangle to Destination
**Objective**: Verify individual object movement to predefined destination

**Steps**:
1. Run `python PRRR_ani.py`
2. Note the Green Triangle destination printed in terminal (e.g., `[x, y, 2.0]`)
3. Click **"Move Triangle"** button

**Expected Result**:
- Robot moves to triangle's initial position
- Picks up triangle (turns red)
- Moves to destination coordinates
- Places triangle at Z=2.0 (or higher if another object is already there)
- Triangle returns to green color

---

### Test Case 2: Move Circle to Destination
**Objective**: Verify individual object movement with different Z-height

**Steps**:
1. Note the Yellow Circle destination printed in terminal (e.g., `[x, y, 4.0]`)
2. Click **"Move Circle"** button

**Expected Result**:
- Robot picks up yellow circle
- Moves to destination coordinates
- Places circle at Z=4.0 (or higher if stacking occurs)
- Circle returns to yellow color

---

### Test Case 3: Move Blue Square to Custom Destination
**Objective**: Test user-defined coordinates with smart stacking

**Steps**:
1. In "Blue Square Control" text box, enter: `2 3 1`
2. Click **"GO"** button

**Expected Result**:
- Robot picks up blue square
- Moves to (2, 3, Z)
- If no object at (2, 3): places at Z=1.0
- If object exists at (2, 3): stacks on top (Z = existing_height + 0.6)

---

### Test Case 4: Stack All 3 Objects at Custom Destination
**Objective**: Verify Master Stack functionality with dynamic stacking

**Steps**:
1. In "Master Stack Control" text box, enter any coordinate within workspace
2. Click **"STACK ALL"** button

**Expected Result**:
- First object: placed at Z=0.0
- Second object: placed at Z=0.6 (on top of first)
- Third object: placed at Z=1.2 (on top of second)
- Terminal prints movement status for each object
- All three objects stacked vertically at (0, 0)

---

### Test Case 5: Destination Outside Workspace (Unreachable)
**Objective**: Verify early validation prevents impossible movements

**Steps**:
1. In "Blue Square Control", enter: `10 10 3` (beyond max reach of 7.5)
2. Click **"GO"** button

**Expected Result**:
- Terminal prints: `"Outside workspace , cannot reach"`
- Robot does NOT pick up the object
- Blue square remains at current position
- No movement occurs

---

### Test Case 6: Dynamic Stacking on Existing Object
**Objective**: Verify smart stacking detects and stacks on pre-existing objects

**Steps**:
1. Move Green Triangle to `-2 0 1`:
   - In "Blue Square Control", enter: `-2 0 1`
   - Click "GO"
2. Move Blue Square to same location `-2 0 1`:
   - Enter: `-2 0 1` again
   - Click "GO"

**Expected Result**:
- First object placed at Z=1.0
- Second object automatically stacked at Z=1.6 (1.0 + 0.6)
- Terminal shows smart Z calculation
- Objects vertically aligned at (-2, 0)

---

### Test Case 7: Skip Objects Already at Target
**Objective**: Verify optimization that skips redundant movements

**Steps**:
1. Move Blue Square to `4 4 0`
2. Use Master Stack with `4 4 0`

**Expected Result**:
- Terminal prints: `"Skipping Blue Square (Already at target)"`
- Blue square is NOT moved
- Other two objects move to (4, 4) and stack

---

### Test Case 8: Reset Simulation
**Objective**: Verify complete system reset

**Steps**:
1. Perform several movements (any test cases above)
2. Click **"RESET SIMULATION"** button

**Expected Result**:
- Terminal prints: `"Resetting..."`
- New random destinations printed for Triangle and Circle
- All objects return to new random starting positions
- Robot returns to home position (all joints at 0)
- Visualization updates to show new configuration

---

### Test Case 9: Workspace Visualization Toggle
**Objective**: Verify workspace envelope display

**Steps**:
1. Check the **"Show Workspace"** checkbox

**Expected Result**:
- Yellow cylindrical surface appears in 3D view
- Cylinder radius = 7.5 units (MAX_REACH)
- Cylinder height = 5.0 units (MAX_Z_HEIGHT)
- Helps visualize reachable workspace

---

### Test Case 10: Master Stack with Z-Offset
**Objective**: Test stacking starting from non-zero height

**Steps**:
1. In "Master Stack Control", enter: `3 3 2`
2. Click **"STACK ALL"**

**Expected Result**:
- First object: placed at Z=2.0
- Second object: placed at Z=2.6
- Third object: placed at Z=3.2
- All objects stacked starting from Z=2.0

---

### Additional Edge Cases

**Test Case 11: Negative Coordinates**
- Input: `-5 -5 1`
- Expected: Movement succeeds if within reach

**Test Case 12: Zero Coordinates**
- Input: `0 0 0`
- Expected: All objects stack at origin

**Test Case 13: Maximum Reach Boundary**
- Input: `7.5 0 2.5` (exactly at max reach)
- Expected: Movement succeeds (boundary case)

**Test Case 14: Sequential Stacking**
- Move objects one-by-one to same location
- Expected: Each subsequent object stacks higher

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
