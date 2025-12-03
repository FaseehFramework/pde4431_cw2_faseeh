import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ============================================================================
# ROBOT PARAMETERS
# ============================================================================
L1 = 3.0  # Link 1 length (vertical)
L2 = 5.0  # Link 2 length (horizontal)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
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

def get_robot_points(theta1, theta2):
    """Calculate joint positions using Forward Kinematics."""
    # Base position (origin)
    P0 = np.array([0, 0, 0, 1])

    # Link 1 transformation: Base to Joint 2
    T01 = dh_transform(a=0, alpha=np.pi/2, d=L1, theta=theta1)
    P1_homogeneous = T01 @ P0
    P1 = P1_homogeneous[:3]

    # Link 2 transformation: Joint 2 to End-effector
    T12 = dh_transform(a=L2, alpha=0, d=0, theta=theta2)
    T02 = T01 @ T12
    P2_homogeneous = T02 @ P0
    P2 = P2_homogeneous[:3]
    
    return np.array([0, 0, 0]), P1, P2

def setup_plot(ax, title):
    """Configure plot limits and labels."""
    ax.set_xlabel('X', fontsize=10, fontweight='bold')
    ax.set_ylabel('Y', fontsize=10, fontweight='bold')
    ax.set_zlabel('Z', fontsize=10, fontweight='bold')
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

def draw_robot(ax, base, joint2, end_effector, label_suffix=""):
    """Draw the robot links and joints."""
    # Draw Link 1 (Base to Joint 2)
    ax.plot([base[0], joint2[0]], 
            [base[1], joint2[1]], 
            [base[2], joint2[2]], 
            'b-', linewidth=4, label=f'Link 1')

    # Draw Link 2 (Joint 2 to End-effector)
    ax.plot([joint2[0], end_effector[0]], 
            [joint2[1], end_effector[1]], 
            [joint2[2], end_effector[2]], 
            'g-', linewidth=4, label=f'Link 2')

    # Plot joints
    ax.scatter([base[0]], [base[1]], [base[2]], 
               color='red', s=100, marker='o', 
               edgecolors='black', linewidths=1, zorder=5)

    ax.scatter([joint2[0]], [joint2[1]], [joint2[2]], 
               color='orange', s=100, marker='o', 
               edgecolors='black', linewidths=1, zorder=5)

    ax.scatter([end_effector[0]], [end_effector[1]], [end_effector[2]], 
               color='purple', s=100, marker='s', 
               label=f'End-effector {label_suffix}', edgecolors='black', linewidths=1, zorder=5)

def set_axes_equal(ax, points):
    """Set 3D plot axes to equal scale."""
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

# ============================================================================
# MAIN EXECUTION
# ============================================================================
if __name__ == "__main__":
    # Setup Figure
    fig = plt.figure(figsize=(16, 8))
    
    # --- LEFT PLOT: FORWARD KINEMATICS ---
    ax1 = fig.add_subplot(121, projection='3d')
    
    # 1. FK Inputs
    fk_theta1 = np.radians(83.82)
    fk_theta2 = np.radians(45.45)
    
    print("="*40)
    print("FORWARD KINEMATICS")
    print("="*40)
    print(f"Input Joint Variables:")
    print(f"  Theta 1: {np.degrees(fk_theta1):.4f} degrees")
    print(f"  Theta 2: {np.degrees(fk_theta2):.4f} degrees")
    
    # 2. Calculate FK
    fk_base, fk_joint2, fk_ee = get_robot_points(fk_theta1, fk_theta2)
    
    print(f"\nCalculated End-Effector Position:")
    print(f"  X: {fk_ee[0]:.4f}")
    print(f"  Y: {fk_ee[1]:.4f}")
    print(f"  Z: {fk_ee[2]:.4f}")
    print("-" * 40)
    
    # Draw FK
    draw_robot(ax1, fk_base, fk_joint2, fk_ee, "(FK)")
    setup_plot(ax1, f'Forward Kinematics\nθ1={np.degrees(fk_theta1):.1f}°, θ2={np.degrees(fk_theta2):.1f}°')
    
    # Add info text
    fk_info = f'Pos: ({fk_ee[0]:.2f}, {fk_ee[1]:.2f}, {fk_ee[2]:.2f})'
    ax1.text2D(0.05, 0.95, fk_info, transform=ax1.transAxes, bbox=dict(facecolor='white', alpha=0.7))

    # Set equal aspect for FK
    all_points_fk = np.array([fk_base, fk_joint2, fk_ee])
    ax1.set_xlim(-8, 8)
    ax1.set_ylim(-8, 8)
    ax1.set_zlim(0, 10)


    # --- RIGHT PLOT: INVERSE KINEMATICS ---
    ax2 = fig.add_subplot(122, projection='3d')
    
    # 3. IK Inputs (Take from FK output)
    target_x = fk_ee[0]
    target_y = fk_ee[1]
    target_z = fk_ee[2]
    
    print("\n" + "="*40)
    print("INVERSE KINEMATICS")
    print("="*40)
    print(f"Input Target Position (from FK):")
    print(f"  X: {target_x:.4f}")
    print(f"  Y: {target_y:.4f}")
    print(f"  Z: {target_z:.4f}")
    
    try:
        # 4. Calculate IK
        ik_theta1, ik_theta2 = inverse_kinematics(target_x, target_y, target_z)
        
        print(f"\nCalculated Joint Variables:")
        print(f"  Theta 1: {np.degrees(ik_theta1):.4f} degrees")
        print(f"  Theta 2: {np.degrees(ik_theta2):.4f} degrees")
        print("="*40)
        
        # Calculate FK from IK results to visualize
        ik_base, ik_joint2, ik_ee = get_robot_points(ik_theta1, ik_theta2)
        
        # Draw IK
        draw_robot(ax2, ik_base, ik_joint2, ik_ee, "(Calc)")
        
        # Draw Target
        ax2.scatter([target_x], [target_y], [target_z], 
                   color='cyan', s=150, marker='x', 
                   label='Target', zorder=10)
        
        setup_plot(ax2, f'Inverse Kinematics\nTarget: ({target_x:.2f}, {target_y:.2f}, {target_z:.2f})')
        
        # Add info text
        ik_info = f'Result:\nθ1={np.degrees(ik_theta1):.1f}°\nθ2={np.degrees(ik_theta2):.1f}°'
        ax2.text2D(0.05, 0.95, ik_info, transform=ax2.transAxes, bbox=dict(facecolor='white', alpha=0.7))
        
        # Set equal aspect for IK
        ax2.set_xlim(-8, 8)
        ax2.set_ylim(-8, 8)
        ax2.set_zlim(0, 10)
        
        ax2.legend()

    except ValueError as e:
        print(f"\nERROR: {e}")
        setup_plot(ax2, "Inverse Kinematics (Error)")
        ax2.text2D(0.5, 0.5, str(e), transform=ax2.transAxes, ha='center', color='red')

    plt.tight_layout()
    plt.show()
