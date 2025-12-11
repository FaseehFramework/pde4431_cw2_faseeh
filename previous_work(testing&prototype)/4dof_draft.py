""" 
DH transformation for 4-DOF R-R-R-P robot 
Configuration: Waist (R) -> Shoulder (R) -> Elbow (R) -> Extension (P)
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def dh_transform(a, alpha, d, theta):
    """Standard (Classic) DH transformation matrix."""
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [ct, -st*ca,  st*sa, a*ct],
        [st,  ct*ca, -ct*sa, a*st],
        [0,      sa,     ca,    d],
        [0,       0,      0,    1]
    ])

# ============================================================================
# ROBOT PARAMETERS & CONFIGURATION
# ============================================================================

# Fixed Link Lengths
L1 = 4.0  # Base height
L2 = 5.0  # Humerus (Upper Arm) length
L3 = 4.0  # Forearm length

# Variable Joint Parameters (The 4 Degrees of Freedom)
# Change these values to see the robot move
theta1 = np.radians(45)   # J1 (Rotational): Base Waist
theta2 = np.radians(60)   # J2 (Rotational): Shoulder
theta3 = np.radians(-60)  # J3 (Rotational): Elbow
d4     = 3.0              # J4 (Prismatic):  Extension length (Variable)

# ============================================================================
# FORWARD KINEMATICS CALCULATION
# ============================================================================

# Base position (origin)
P0 = np.array([0, 0, 0, 1])

# --- Transform 0->1 (Base to Shoulder) ---
# Rotate theta1 around Z, translate L1 up Z, rotate X by 90 deg
T01 = dh_transform(a=0, alpha=np.pi/2, d=L1, theta=theta1)

# --- Transform 1->2 (Shoulder to Elbow) ---
# Rotate theta2 around Z, translate L2 along X
T12 = dh_transform(a=L2, alpha=0, d=0, theta=theta2)

# --- Transform 2->3 (Elbow to Wrist/Tool Base) ---
# Rotate theta3 around Z, translate L3 along X, rotate X by -90 deg to orient Z for the slide
T23 = dh_transform(a=L3, alpha=-np.pi/2, d=0, theta=theta3)

# --- Transform 3->4 (Prismatic Extension) ---
# No rotation (theta=0), Translate d4 along Z (Prismatic action)
T34 = dh_transform(a=0, alpha=0, d=d4, theta=0) 

# Calculate Cumulative Transformations (Base to each Frame)
T02 = T01 @ T12
T03 = T02 @ T23
T04 = T03 @ T34 # Final Matrix (Base to End Effector)

# Extract Positions for Plotting (Extracting [x,y,z] from homogeneous vectors)
pos_base = P0[:3]
pos_j2   = (T01 @ P0)[:3]
pos_j3   = (T02 @ P0)[:3]
pos_j4   = (T03 @ P0)[:3] # Start of prismatic joint
pos_ee   = (T04 @ P0)[:3] # End effector (Tip)

# Group points for easy looping
points = [pos_base, pos_j2, pos_j3, pos_j4, pos_ee]

# Print Final Coordinate
print(f"End-effector position (x,y,z): ({pos_ee[0]:.4f}, {pos_ee[1]:.4f}, {pos_ee[2]:.4f})")

# ============================================================================
# 3D VISUALIZATION
# ============================================================================

fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# 1. Plot Link Segments
# Link 1 (Vertical Base)
ax.plot([points[0][0], points[1][0]], 
        [points[0][1], points[1][1]], 
        [points[0][2], points[1][2]], 
        color='black', linewidth=8, label='Link 1 (Base)')

# Link 2 (Upper Arm)
ax.plot([points[1][0], points[2][0]], 
        [points[1][1], points[2][1]], 
        [points[1][2], points[2][2]], 
        color='blue', linewidth=6, label='Link 2 (Arm)')

# Link 3 (Forearm)
ax.plot([points[2][0], points[3][0]], 
        [points[2][1], points[3][1]], 
        [points[2][2], points[3][2]], 
        color='green', linewidth=6, label='Link 3 (Forearm)')

# Link 4 (Prismatic Extension) - Dashed to indicate sliding nature
ax.plot([points[3][0], points[4][0]], 
        [points[3][1], points[4][1]], 
        [points[3][2], points[4][2]], 
        color='red', linewidth=4, linestyle='--', label=f'Link 4 (Prismatic d={d4})')

# 2. Plot Joints
# Base (Joint 1)
ax.scatter(points[0][0], points[0][1], points[0][2], s=200, c='gray', marker='o')
# Shoulder (Joint 2)
ax.scatter(points[1][0], points[1][1], points[1][2], s=150, c='orange', marker='o')
# Elbow (Joint 3)
ax.scatter(points[2][0], points[2][1], points[2][2], s=150, c='orange', marker='o')
# Prismatic Base (Start of Extension)
ax.scatter(points[3][0], points[3][1], points[3][2], s=100, c='purple', marker='s')
# End Effector
ax.scatter(points[4][0], points[4][1], points[4][2], s=200, c='red', marker='*', label='End Effector')

# 3. Draw Coordinate Frame at Base
frame_scale = 2.0
ax.quiver(0, 0, 0, frame_scale, 0, 0, color='r', alpha=0.5) # X
ax.quiver(0, 0, 0, 0, frame_scale, 0, color='g', alpha=0.5) # Y
ax.quiver(0, 0, 0, 0, 0, frame_scale, color='b', alpha=0.5) # Z

# 4. Plot Settings
ax.set_xlabel('X Axis')
ax.set_ylabel('Y Axis')
ax.set_zlabel('Z Axis')
title_str = f'4-DOF (R-R-R-P) Robot\nAngles: [{np.degrees(theta1):.0f}°, {np.degrees(theta2):.0f}°, {np.degrees(theta3):.0f}°], Ext: {d4}'
ax.set_title(title_str, fontsize=14, fontweight='bold')

# 5. Set Aspect Ratio (Equal box)
all_x = [p[0] for p in points]
all_y = [p[1] for p in points]
all_z = [p[2] for p in points]
max_range = np.array([np.ptp(all_x), np.ptp(all_y), np.ptp(all_z)]).max() / 2.0
mid_x, mid_y, mid_z = np.mean(all_x), np.mean(all_y), np.mean(all_z)

ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, mid_y + max_range)
ax.set_zlim(mid_z - max_range, mid_z + max_range)

ax.legend()
plt.tight_layout()
plt.show()