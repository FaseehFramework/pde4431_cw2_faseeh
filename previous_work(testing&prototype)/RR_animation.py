import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

# ============================================================================
# ROBOT CONFIGURATION
# ============================================================================
L1 = 3.0  # Link 1 (Vertical)
L2 = 5.0  # Link 2 (Horizontal)
STEP_SIZE_DEG = 5.0  # Movement increment per frame in degrees

class RobotArmSimulator:
    def __init__(self):
        # Initialize robot at "Home" position (0, 0)
        # Note: (0,0,0) cartesian is unreachable for this robot geometry
        self.current_theta1 = 0.0
        self.current_theta2 = 0.0
        
        # Setup the visualization
        self.fig = plt.figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.running = True
        
        # Setup event listener for 'q' to quit
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)

    def on_key(self, event):
        if event.key == 'q':
            print("\n[System] Quitting simulation...")
            self.running = False
            plt.close(self.fig)

    def dh_transform(self, a, alpha, d, theta):
        """Standard DH Transformation Matrix"""
        ct, st = np.cos(theta), np.sin(theta)
        ca, sa = np.cos(alpha), np.sin(alpha)
        return np.array([
            [ct, -st*ca,  st*sa, a*ct],
            [st,  ct*ca, -ct*sa, a*st],
            [0,      sa,     ca,    d],
            [0,       0,      0,    1]
        ])

    def forward_kinematics(self, theta1, theta2):
        """Calculates 3D points for Base, Joint2, and End-Effector"""
        # Base (Origin)
        P0 = np.array([0, 0, 0, 1])

        # Link 1: Base -> Joint 2
        T01 = self.dh_transform(a=0, alpha=np.pi/2, d=L1, theta=theta1)
        P1 = (T01 @ P0)[:3]

        # Link 2: Joint 2 -> End Effector
        T12 = self.dh_transform(a=L2, alpha=0, d=0, theta=theta2)
        P2 = (T01 @ T12 @ P0)[:3]

        return np.array([0, 0, 0]), P1, P2

    def inverse_kinematics(self, x, y, z):
        """
        Calculates joint angles. 
        If target is out of reach, projects it onto the reachable sphere.
        """
        # Vector from Shoulder (0,0,L1) to Target (x,y,z)
        dx = x
        dy = y
        dz = z - L1
        
        # Calculate distance from shoulder
        dist = np.sqrt(dx**2 + dy**2 + dz**2)
        
        # 1. REACHABILITY CHECK & PROJECTION
        # If the target is not on the sphere surface (tolerance 0.01), project it.
        if not np.isclose(dist, L2, atol=0.01):
            scale = L2 / dist
            print(f"[Smart Assist] Target ({x}, {y}, {z}) unreachable (Dist={dist:.2f}).")
            
            # Project vector to be length L2
            dx *= scale
            dy *= scale
            dz *= scale
            
            # Recalculate actual reachable coordinates
            x, y = dx, dy
            z = dz + L1
            print(f"               Moving to closest point: ({x:.2f}, {y:.2f}, {z:.2f})")

        # 2. SOLVE ANGLES (Standard IK)
        # Pitch (Theta 2)
        # z_local = L2 * sin(theta2) -> dz = L2 * sin(theta2)
        # Therefore sin(theta2) = dz / L2
        sin_theta2 = dz / L2
        
        # Clamp value to [-1, 1] to handle floating point noise
        sin_theta2 = max(-1.0, min(1.0, sin_theta2))
        theta2 = np.arcsin(sin_theta2)

        # Yaw (Theta 1)
        # Use atan2 for full circle coverage
        theta1 = np.arctan2(y, x)

        return theta1, theta2

    def interpolate_joint_space(self, target_t1, target_t2):
        """Generates a list of intermediate angles (Trajectory)"""
        # Calculate angular distance
        diff_t1 = target_t1 - self.current_theta1
        diff_t2 = target_t2 - self.current_theta2
        
        # Determine number of steps based on the largest joint movement
        max_diff_deg = np.degrees(max(abs(diff_t1), abs(diff_t2)))
        steps = int(max_diff_deg / STEP_SIZE_DEG)
        
        if steps < 2: steps = 2 # Ensure at least start and end
        
        # Generate smooth path
        t1_path = np.linspace(self.current_theta1, target_t1, steps)
        t2_path = np.linspace(self.current_theta2, target_t2, steps)
        
        return zip(t1_path, t2_path)

    def draw_frame(self, t1, t2, target_pos=None):
        """Render a single animation frame"""
        self.ax.cla() # Clear previous frame
        
        # Define World Frame (Fixed Ranges)
        self.ax.set_xlim(-10, 10)
        self.ax.set_ylim(-10, 10)
        self.ax.set_zlim(0, 10)
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_title(f'Robot Simulation\nJoints: {np.degrees(t1):.1f}°, {np.degrees(t2):.1f}°')
        
        # Get robot points using FK
        base, joint2, ee = self.forward_kinematics(t1, t2)
        
        # Draw Links
        self.ax.plot([base[0], joint2[0]], [base[1], joint2[1]], [base[2], joint2[2]], 'b-', linewidth=4)
        self.ax.plot([joint2[0], ee[0]], [joint2[1], ee[1]], [joint2[2], ee[2]], 'g-', linewidth=4)
        
        # Draw Joints
        self.ax.scatter(*base, c='k', s=100)
        self.ax.scatter(*joint2, c='orange', s=100)
        self.ax.scatter(*ee, c='purple', s=100)
        
        # Draw Target Ghost
        if target_pos is not None:
            self.ax.scatter(*target_pos, c='r', marker='x', s=150, label='Target')
            self.ax.legend()
            
        plt.draw()
        plt.pause(0.01) # Small pause to allow GUI to update

    def run(self):
        print("="*50)
        print("ROBOT SIMULATION STARTED")
        print(f"Robot Geometry: L1={L1}, L2={L2}")
        print("Commands:")
        print("  - Enter X, Y, Z coordinates to move.")
        print("  - Press 'q' in the console (or close plot) to quit.")
        print("="*50)

        # Initial Draw
        self.draw_frame(self.current_theta1, self.current_theta2)
        plt.show(block=False)

        while self.running:
            try:
                # 1. Get User Input
                user_input = input("\nEnter Target (x y z) or 'q' to quit: ")
                if user_input.lower() == 'q':
                    self.running = False
                    break
                
                try:
                    vals = [float(v) for v in user_input.split(',')] if ',' in user_input else [float(v) for v in user_input.split()]
                    if len(vals) != 3:
                        print("Error: Please enter exactly 3 numbers (x y z).")
                        continue
                    tx, ty, tz = vals
                except ValueError:
                    print("Error: Invalid number format.")
                    continue

                # 2. Calculate Inverse Kinematics
                try:
                    print(f"Calculating path to ({tx}, {ty}, {tz})...")
                    target_t1, target_t2 = self.inverse_kinematics(tx, ty, tz)
                    print(f"Solution: J1={np.degrees(target_t1):.2f}°, J2={np.degrees(target_t2):.2f}°")
                except ValueError as e:
                    print(f"Cannot Move: {e}")
                    continue

                # 3. Interpolate Path (Joint Space)
                trajectory = list(self.interpolate_joint_space(target_t1, target_t2))
                print(f"Moving... ({len(trajectory)} frames)")

                # 4. Animate Movement
                for t1, t2 in trajectory:
                    if not self.running: break
                    self.draw_frame(t1, t2, target_pos=(tx, ty, tz))
                
                # 5. Update Internal State
                self.current_theta1 = target_t1
                self.current_theta2 = target_t2
                print("Movement Complete.")

            except KeyboardInterrupt:
                self.running = False

if __name__ == "__main__":
    sim = RobotArmSimulator()
    sim.run()