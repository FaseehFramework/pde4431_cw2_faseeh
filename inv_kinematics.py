import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

L1 = 3.0  # Vertical Link
L2 = 5.0  # Horizontal Link

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

def inverse_kinematics(x, y, z):
    """
    Input: Target coordinates (x, y, z)
    Output: Joint angles (theta1, theta2) in radians
    """
    
    # --- Step A: Solve for Theta 2 (Pitch) ---
    # We know: z = L1 + L2 * sin(theta2)
    sin_theta2 = (z - L1) / L2
    
    # Safety Check: Is the target reachable vertically?
    if abs(sin_theta2) > 1.0:
        raise ValueError(f"Target Z={z} is out of reach for this robot.")

    # Calculate theta2
    theta2 = np.arcsin(sin_theta2)

    # --- Step B: Solve for Theta 1 (Base Yaw) ---
    # If theta2 puts the arm straight up (90 deg), x and y must be 0.
    if np.isclose(np.cos(theta2), 0):
        theta1 = 0.0 # Singularity: Base angle is arbitrary
    else:
        theta1 = np.arctan2(y, x)

    return theta1, theta2

target_x = 1.9
target_y = 3.4
target_z = 6.0

print(f"--- TARGET: ({target_x}, {target_y}, {target_z}) ---\n")

try:
    # 1. Calculate Inverse Kinematics
    theta1, theta2 = inverse_kinematics(target_x, target_y, target_z)
    
    print(f"Calculated Joint Angles:")
    print(f"Theta 1: {np.degrees(theta1):.4f} degrees")
    print(f"Theta 2: {np.degrees(theta2):.4f} degrees")
    print("-" * 30)

    # 2. Compute Joint Positions for Plotting (Forward Kinematics logic)
    # Base position (origin)
    P0 = np.array([0, 0, 0, 1])

    # Link 1 transformation: Base to Joint 2
    T01 = dh_transform(a=0, alpha=np.pi/2, d=L1, theta=theta1)
    P1 = (T01 @ P0)[:3]  # Joint 2 position

    # Link 2 transformation: Joint 2 to End-effector
    T12 = dh_transform(a=L2, alpha=0, d=0, theta=theta2)
    T02 = T01 @ T12      # Total transform
    P2 = (T02 @ P0)[:3]  # End-effector position

    # 3. Visualization
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Points
    base = np.array([0, 0, 0])
    joint2 = P1
    end_effector = P2

    # Draw Link 1 (Base to Joint 2)
    ax.plot([base[0], joint2[0]], 
            [base[1], joint2[1]], 
            [base[2], joint2[2]], 
            'b-', linewidth=6, label=f'Link 1 (L={L1})')

    # Draw Link 2 (Joint 2 to End-effector)
    ax.plot([joint2[0], end_effector[0]], 
            [joint2[1], end_effector[1]], 
            [joint2[2], end_effector[2]], 
            'g-', linewidth=6, label=f'Link 2 (L={L2})')

    # Plot joints
    ax.scatter([base[0]], [base[1]], [base[2]], 
               color='red', s=200, marker='o', 
               label='Base', edgecolors='black', zorder=5)

    ax.scatter([joint2[0]], [joint2[1]], [joint2[2]], 
               color='orange', s=200, marker='o', 
               label='Joint 2', edgecolors='black', zorder=5)

    # Plot target vs Actual End-effector
    ax.scatter([end_effector[0]], [end_effector[1]], [end_effector[2]], 
               color='purple', s=200, marker='s', 
               label='Calculated Position', edgecolors='black', zorder=5)
    
    ax.scatter([target_x], [target_y], [target_z], 
               color='cyan', s=100, marker='x', 
               label='Target Input', zorder=10)

    # Formatting
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Inverse Kinematics Result\nTarget: ({target_x}, {target_y}, {target_z})')

    # Set equal aspect ratio
    all_points = np.array([base, joint2, end_effector, [target_x, target_y, target_z]])
    max_range = np.array([np.ptp(all_points[:, 0]),
                          np.ptp(all_points[:, 1]),
                          np.ptp(all_points[:, 2])]).max() / 2.0
    max_range = max(max_range, 2.0) * 1.2
    
    mid_x, mid_y, mid_z = all_points[:, 0].mean(), all_points[:, 1].mean(), all_points[:, 2].mean()

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    ax.set_box_aspect([1, 1, 1])

    ax.legend()
    plt.tight_layout()
    plt.show()

except ValueError as e:
    print(f"\nERROR: {e}")