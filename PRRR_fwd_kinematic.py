""" DH transformation for 4-DOF PRRR robot (Prismatic + 3 Revolute joints) """
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

# Robot parameters (link lengths)
L1 = 2.5  # Prismatic joint displacement (variable, this is the initial value)
L2 = 3.0  # Link 2 length
L3 = 2.5  # Link 3 length
L4 = 2.0  # Link 4 length (end-effector)

# Joint variables (change these to test different configurations)
d1 = 2.5              # Prismatic joint displacement (vertical, along z-axis)
theta1 = np.radians(0)    # Rotation about z0 (base rotation)
theta2 = np.radians(30)   # Second joint rotation (about z1)
theta3 = np.radians(45)   # Third joint rotation (about z2)
theta4 = np.radians(-20)  # Fourth joint rotation (about z3)

# Base position (origin)
P0 = np.array([0, 0, 0, 1])  # x=0, y=0, z=0, 1

# ============================================================================
# DH PARAMETER TABLE FOR PRRR MANIPULATOR
# ============================================================================
# Joint | a_i-1 | alpha_i-1 | d_i | theta_i
# ------|-------|-----------|-----|--------
#   1   |   0   |     0     | d1  | theta1  (Prismatic - vertical motion)
#   2   |   0   |   π/2     |  0  | theta2  (Revolute)
#   3   |  L2   |     0     |  0  | theta3  (Revolute)
#   4   |  L3   |     0     |  0  | theta4  (Revolute)
#   EE  |  L4   |     0     |  0  |    0    (End-effector)
# ============================================================================

print("="*70)
print("PRRR MANIPULATOR - FORWARD KINEMATICS")
print("="*70)
print(f"\nLink Lengths:")
print(f"  L2 = {L2} (Link 2)")
print(f"  L3 = {L3} (Link 3)")
print(f"  L4 = {L4} (Link 4 - End-effector)")
print(f"\nJoint Configuration:")
print(f"  d1 (Prismatic) = {d1}")
print(f"  θ1 = {np.degrees(theta1):.2f}°")
print(f"  θ2 = {np.degrees(theta2):.2f}°")
print(f"  θ3 = {np.degrees(theta3):.2f}°")
print(f"  θ4 = {np.degrees(theta4):.2f}°")
print("="*70)
print()

# Joint 1 transformation: Base to Joint 2 (Prismatic + Rotation)
T01 = dh_transform(a=0, alpha=0, d=d1, theta=theta1)
print("Transformation T01 (Base to Joint 2 - Prismatic):")
print(T01)
print()

# Position of Joint 2
P1_homogeneous = T01 @ P0
P1 = P1_homogeneous[:3]
print(f"Joint 2 position: ({P1[0]:.4f}, {P1[1]:.4f}, {P1[2]:.4f})")
print()

# Joint 2 transformation: Joint 2 to Joint 3
T12 = dh_transform(a=0, alpha=np.pi/2, d=0, theta=theta2)
print("Transformation T12 (Joint 2 to Joint 3):")
print(T12)
print()

# Total transformation to Joint 3
T02 = T01 @ T12
P2_homogeneous = T02 @ P0
P2 = P2_homogeneous[:3]
print(f"Joint 3 position: ({P2[0]:.4f}, {P2[1]:.4f}, {P2[2]:.4f})")
print()

# Joint 3 transformation: Joint 3 to Joint 4
T23 = dh_transform(a=L2, alpha=0, d=0, theta=theta3)
print("Transformation T23 (Joint 3 to Joint 4):")
print(T23)
print()

# Total transformation to Joint 4
T03 = T02 @ T23
P3_homogeneous = T03 @ P0
P3 = P3_homogeneous[:3]
print(f"Joint 4 position: ({P3[0]:.4f}, {P3[1]:.4f}, {P3[2]:.4f})")
print()

# Joint 4 transformation: Joint 4 to End-effector
T34 = dh_transform(a=L3, alpha=0, d=0, theta=theta4)
print("Transformation T34 (Joint 4 to End-effector):")
print(T34)
print()

# Total transformation: Base to End-effector
T04 = T03 @ T34
print("Total Transformation T04 (Base to End-effector):")
print(T04)
print()

# Position of End-effector (before final link extension)
P4_homogeneous = T04 @ P0
P4 = P4_homogeneous[:3]
print(f"End-effector base position: ({P4[0]:.4f}, {P4[1]:.4f}, {P4[2]:.4f})")

# Final end-effector position (with L4 extension)
T4E = dh_transform(a=L4, alpha=0, d=0, theta=0)
T0E = T04 @ T4E
PE_homogeneous = T0E @ P0
PE = PE_homogeneous[:3]

print("="*70)
print(f"FINAL END-EFFECTOR POSITION: ({PE[0]:.4f}, {PE[1]:.4f}, {PE[2]:.4f})")
print("="*70)
print()

# ============================================================================
# 3D VISUALIZATION
# ============================================================================

fig = plt.figure(figsize=(14, 12))
ax = fig.add_subplot(111, projection='3d')

# Joint positions
base = np.array([0, 0, 0])
joint2 = P1
joint3 = P2
joint4 = P3
joint5 = P4
end_effector = PE

# Draw Prismatic Link (Base to Joint 2)
ax.plot([base[0], joint2[0]], 
        [base[1], joint2[1]], 
        [base[2], joint2[2]], 
        'b-', linewidth=8, label=f'Prismatic (d={d1})', alpha=0.8)

# Draw Link 2 (Joint 2 to Joint 3)
ax.plot([joint2[0], joint3[0]], 
        [joint2[1], joint3[1]], 
        [joint2[2], joint3[2]], 
        'g-', linewidth=6, label=f'Link 2 (L={L2})', alpha=0.8)

# Draw Link 3 (Joint 3 to Joint 4)
ax.plot([joint3[0], joint4[0]], 
        [joint3[1], joint4[1]], 
        [joint3[2], joint4[2]], 
        'orange', linewidth=6, label=f'Link 3 (L={L3})', alpha=0.8)

# Draw Link 4 (Joint 4 to Joint 5)
ax.plot([joint4[0], joint5[0]], 
        [joint4[1], joint5[1]], 
        [joint4[2], joint5[2]], 
        'purple', linewidth=6, label=f'Link 4 (L={L4})', alpha=0.8)

# Draw End-effector extension (Joint 5 to End-effector)
ax.plot([joint5[0], end_effector[0]], 
        [joint5[1], end_effector[1]], 
        [joint5[2], end_effector[2]], 
        'red', linewidth=5, label=f'End-effector (L={L4})', alpha=0.8, linestyle='--')

# Plot joints
ax.scatter([base[0]], [base[1]], [base[2]], 
           color='red', s=250, marker='o', 
           label='Base (Joint 1)', edgecolors='black', linewidths=2, zorder=5)

ax.scatter([joint2[0]], [joint2[1]], [joint2[2]], 
           color='cyan', s=200, marker='o', 
           label='Joint 2', edgecolors='black', linewidths=2, zorder=5)

ax.scatter([joint3[0]], [joint3[1]], [joint3[2]], 
           color='lime', s=200, marker='o', 
           label='Joint 3', edgecolors='black', linewidths=2, zorder=5)

ax.scatter([joint4[0]], [joint4[1]], [joint4[2]], 
           color='orange', s=200, marker='o', 
           label='Joint 4', edgecolors='black', linewidths=2, zorder=5)

ax.scatter([end_effector[0]], [end_effector[1]], [end_effector[2]], 
           color='darkred', s=300, marker='s', 
           label='End-effector', edgecolors='black', linewidths=2, zorder=5)

# Draw coordinate frame at base
frame_scale = 1.0
ax.quiver(0, 0, 0, frame_scale, 0, 0, color='red', arrow_length_ratio=0.3, linewidth=2.5, alpha=0.7)
ax.quiver(0, 0, 0, 0, frame_scale, 0, color='green', arrow_length_ratio=0.3, linewidth=2.5, alpha=0.7)
ax.quiver(0, 0, 0, 0, 0, frame_scale, color='blue', arrow_length_ratio=0.3, linewidth=2.5, alpha=0.7)

# Add axis labels at the arrows
ax.text(frame_scale*1.2, 0, 0, 'X', color='red', fontsize=12, fontweight='bold')
ax.text(0, frame_scale*1.2, 0, 'Y', color='green', fontsize=12, fontweight='bold')
ax.text(0, 0, frame_scale*1.2, 'Z', color='blue', fontsize=12, fontweight='bold')

# Labels and title
ax.set_xlabel('X', fontsize=12, fontweight='bold')
ax.set_ylabel('Y', fontsize=12, fontweight='bold')
ax.set_zlabel('Z', fontsize=12, fontweight='bold')
ax.set_title(f'4-DOF PRRR Robot Manipulator\nd1={d1}, θ1={np.degrees(theta1):.1f}°, θ2={np.degrees(theta2):.1f}°, θ3={np.degrees(theta3):.1f}°, θ4={np.degrees(theta4):.1f}°', 
             fontsize=14, fontweight='bold')

# Set equal aspect ratio
all_points = np.array([base, joint2, joint3, joint4, joint5, end_effector])
max_range = np.array([np.ptp(all_points[:, 0]),
                      np.ptp(all_points[:, 1]),
                      np.ptp(all_points[:, 2])]).max() / 2.0

# Add some padding
max_range = max(max_range, 2.0) * 1.3

mid_x = all_points[:, 0].mean()
mid_y = all_points[:, 1].mean()
mid_z = all_points[:, 2].mean()

ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, mid_y + max_range)
ax.set_zlim(mid_z - max_range, mid_z + max_range)
ax.set_box_aspect([1, 1, 1])

# Grid and legend
ax.grid(True, alpha=0.3)
ax.legend(loc='upper left', fontsize=9, ncol=2)

# Add text box with info
info_text = (f'Robot Configuration:\n'
             f'Prismatic: d1 = {d1}\n'
             f'L2 = {L2}, L3 = {L3}, L4 = {L4}\n'
             f'θ1 = {np.degrees(theta1):.1f}°\n'
             f'θ2 = {np.degrees(theta2):.1f}°\n'
             f'θ3 = {np.degrees(theta3):.1f}°\n'
             f'θ4 = {np.degrees(theta4):.1f}°\n'
             f'End-effector: ({PE[0]:.2f}, {PE[1]:.2f}, {PE[2]:.2f})')

ax.text2D(0.02, 0.98, info_text, transform=ax.transAxes, 
          fontsize=9, verticalalignment='top',
          bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.9))

plt.tight_layout()
plt.show()

print("*** Visualization complete! ***")
