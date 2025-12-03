import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

# ============================================================================
# ROBOT PARAMETERS (PRRR CONFIGURATION)
# ============================================================================
MAX_Z_HEIGHT = 5.0   # J1 Limit (Vertical rail length)
L_SHOULDER = 3.5     # R1 Length
L_ELBOW = 2.5        # R2 Length
L_WRIST = 1.5        # R3 Length

STEP_SIZE_ANG = 5.0  # Degrees per frame
STEP_SIZE_LIN = 0.2  # Units per frame (for Z-axis)

class PRRR_Robot:
    def __init__(self):
        # Initial State: Home Position
        # d1 (Height), theta1 (Shoulder), theta2 (Elbow), theta3 (Wrist)
        self.d1 = 0.0
        self.theta1 = 0.0
        self.theta2 = 0.0
        self.theta3 = 0.0
        
        self.fig = plt.figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.running = True
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)

    def on_key(self, event):
        if event.key == 'q':
            self.running = False
            plt.close(self.fig)

    def forward_kinematics(self, d1, th1, th2, th3):
        """
        Calculates the 3D coordinates of all joints.
        Chain: Base -> Lift(J1) -> Shoulder(J2) -> Elbow(J3) -> Wrist(J4) -> Tool
        """
        # 1. Base of the Lift (Fixed at Origin)
        p_base = np.array([0, 0, 0])
        
        # 2. Top of the Lift (The Carriage) - determined by d1
        p_lift = np.array([0, 0, d1])
        
        # 3. Shoulder Joint (Same as Lift position, just the start of the arm)
        p_shoulder = p_lift
        
        # 4. Elbow Joint (Rotation th1)
        # x = L1 * cos(th1)
        # y = L1 * sin(th1)
        p_elbow = p_shoulder + np.array([
            L_SHOULDER * np.cos(th1),
            L_SHOULDER * np.sin(th1),
            0
        ])
        
        # 5. Wrist Joint (Rotation th1 + th2)
        # Note: Angles sum up because they are relative
        global_angle_2 = th1 + th2
        p_wrist = p_elbow + np.array([
            L_ELBOW * np.cos(global_angle_2),
            L_ELBOW * np.sin(global_angle_2),
            0
        ])
        
        # 6. End Effector (Rotation th1 + th2 + th3)
        global_angle_3 = global_angle_2 + th3
        p_ee = p_wrist + np.array([
            L_WRIST * np.cos(global_angle_3),
            L_WRIST * np.sin(global_angle_3),
            0
        ])
        
        return p_base, p_lift, p_elbow, p_wrist, p_ee

    def inverse_kinematics(self, x, y, z, target_phi_deg=0):
        """
        Solves IK for PRRR.
        Strategy: Decouple Position and Orientation.
        1. Z is solved directly by J1.
        2. We need the Wrist Joint to be at a specific (Wx, Wy) so that
           the End-Effector reaches (x, y) with angle phi.
        """
        target_phi = np.radians(target_phi_deg)

        # --- STEP 1: SOLVE Z (J1) ---
        if not (0 <= z <= MAX_Z_HEIGHT):
            raise ValueError(f"Target Z={z} is out of vertical range [0, {MAX_Z_HEIGHT}]")
        d1 = z

        # --- STEP 2: CALCULATE WRIST CENTER ---
        # We want the tool tip at (x,y).
        # We know the tool length is L_WRIST.
        # We know the tool angle is target_phi.
        # So, calculate where the wrist joint MUST be.
        wx = x - L_WRIST * np.cos(target_phi)
        wy = y - L_WRIST * np.sin(target_phi)

        # --- STEP 3: SOLVE 2-LINK PLANAR ARM (For Wx, Wy) ---
        # This is standard trigonometry (Law of Cosines)
        # Distance from shoulder to wrist center
        r_sq = wx**2 + wy**2
        r = np.sqrt(r_sq)
        
        # Reachability Check (Planar)
        max_reach = L_SHOULDER + L_ELBOW
        min_reach = abs(L_SHOULDER - L_ELBOW)
        
        if r > max_reach:
            raise ValueError(f"Target is too far ({r:.2f} > {max_reach}).")
        if r < min_reach:
            raise ValueError(f"Target is too close to base ({r:.2f} < {min_reach}).")

        # Law of Cosines for Elbow Angle (theta2)
        # r^2 = L1^2 + L2^2 - 2*L1*L2*cos(180 - theta2)
        # cos(theta2) = (r^2 - L1^2 - L2^2) / (2*L1*L2)
        cos_theta2 = (r_sq - L_SHOULDER**2 - L_ELBOW**2) / (2 * L_SHOULDER * L_ELBOW)
        cos_theta2 = np.clip(cos_theta2, -1.0, 1.0) # Numerical safety
        
        # Two solutions for elbow (Elbow Up / Elbow Down). We pick one.
        theta2 = np.arccos(cos_theta2) # Positive solution (Elbow Left/Up)

        # Solve for Shoulder Angle (theta1)
        # Angle to the wrist center
        alpha = np.arctan2(wy, wx)
        # Angle offset due to geometry
        # Law of Sines or Cosines again
        beta = np.arctan2(L_ELBOW * np.sin(theta2), L_SHOULDER + L_ELBOW * np.cos(theta2))
        
        theta1 = alpha - beta

        # --- STEP 4: SOLVE WRIST (J4/Theta3) ---
        # We want Global Angle = theta1 + theta2 + theta3
        # So theta3 = Global Angle - (theta1 + theta2)
        theta3 = target_phi - (theta1 + theta2)

        return d1, theta1, theta2, theta3

    def interpolate_path(self, target_d1, target_t1, target_t2, target_t3):
        """Generates synchronized joint space trajectory."""
        # Calculate deltas
        delta_d1 = target_d1 - self.d1
        delta_t1 = target_t1 - self.theta1
        delta_t2 = target_t2 - self.theta2
        delta_t3 = target_t3 - self.theta3

        # Determine number of frames based on the slowest joint
        steps_lin = abs(delta_d1) / STEP_SIZE_LIN
        steps_ang = np.degrees(max(abs(delta_t1), abs(delta_t2), abs(delta_t3))) / STEP_SIZE_ANG
        
        steps = int(max(steps_lin, steps_ang))
        if steps < 5: steps = 5 # Minimum frames for smoothness

        # Generate Linspaces
        d1_path = np.linspace(self.d1, target_d1, steps)
        t1_path = np.linspace(self.theta1, target_t1, steps)
        t2_path = np.linspace(self.theta2, target_t2, steps)
        t3_path = np.linspace(self.theta3, target_t3, steps)

        return zip(d1_path, t1_path, t2_path, t3_path)

    def draw_scene(self, d1, t1, t2, t3, target_pos=None):
        self.ax.cla()
        
        # 1. Coordinate System Setup
        self.ax.set_xlim(-8, 8)
        self.ax.set_ylim(-8, 8)
        self.ax.set_zlim(0, 6)
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_title(f"PRRR Robot\nZ={d1:.2f}, J1={np.degrees(t1):.0f}°, J2={np.degrees(t2):.0f}°, J3={np.degrees(t3):.0f}°")

        # 2. Get Points
        base, lift, elbow, wrist, ee = self.forward_kinematics(d1, t1, t2, t3)

        # 3. Draw Robot
        # Vertical Rail (The "P" joint track)
        self.ax.plot([0, 0], [0, 0], [0, MAX_Z_HEIGHT], 'k--', linewidth=1, alpha=0.5)
        
        # Link 1: Lift Column (Base -> Carriage)
        self.ax.plot([base[0], lift[0]], [base[1], lift[1]], [base[2], lift[2]], 
                     'k-', linewidth=6, label='Lift (Z)')
        
        # Link 2: Shoulder (Carriage -> Elbow)
        self.ax.plot([lift[0], elbow[0]], [lift[1], elbow[1]], [lift[2], elbow[2]], 
                     'b-', linewidth=4, label='Shoulder')
        
        # Link 3: Forearm (Elbow -> Wrist)
        self.ax.plot([elbow[0], wrist[0]], [elbow[1], wrist[1]], [elbow[2], wrist[2]], 
                     'g-', linewidth=4, label='Forearm')
        
        # Link 4: Hand (Wrist -> EE)
        self.ax.plot([wrist[0], ee[0]], [wrist[1], ee[1]], [wrist[2], ee[2]], 
                     'r-', linewidth=3, label='Gripper')

        # Joints
        self.ax.scatter(*lift, s=100, c='k', marker='s')     # Prismatic Carriage
        self.ax.scatter(*elbow, s=80, c='b')                 # Shoulder Joint
        self.ax.scatter(*wrist, s=80, c='g')                 # Elbow Joint
        self.ax.scatter(*ee, s=60, c='r', marker='v')        # Tool Tip

        # Target Marker
        if target_pos:
            self.ax.scatter(*target_pos, c='m', marker='x', s=100, label='Goal')

        plt.draw()
        plt.pause(0.01)

    def run(self):
        print("="*50)
        print("PRRR ROBOT CONTROL (SCARA Type)")
        print(f"Dims: Lift={MAX_Z_HEIGHT}, Arm={L_SHOULDER}+{L_ELBOW}+{L_WRIST}")
        print("Feature: Auto-orientation (Gripper stays aligned with global axes)")
        print("="*50)

        # Initial Render
        self.draw_scene(self.d1, self.theta1, self.theta2, self.theta3)
        plt.show(block=False)

        while self.running:
            try:
                raw = input("\nEnter Target (x y z [phi]): ")
                if raw.lower() == 'q': break
                
                parts = [float(x) for x in raw.replace(',', ' ').split()]
                
                # Parse Input
                x, y, z = parts[0], parts[1], parts[2]
                phi = parts[3] if len(parts) > 3 else 0.0 # Default to 0 orientation if not given
                
                # Calculate IK
                print(f"Planning move to ({x}, {y}, {z}) with Angle {phi}°...")
                try:
                    td1, tt1, tt2, tt3 = self.inverse_kinematics(x, y, z, phi)
                except ValueError as e:
                    print(f"Error: {e}")
                    continue

                # Generate Path
                path = list(self.interpolate_path(td1, tt1, tt2, tt3))
                print(f"Executing trajectory ({len(path)} frames)...")

                # Animate
                for d, t1, t2, t3 in path:
                    if not self.running: break
                    self.draw_scene(d, t1, t2, t3, target_pos=(x,y,z))
                
                # Update State
                self.d1, self.theta1, self.theta2, self.theta3 = td1, tt1, tt2, tt3
                print("Done.")

            except (ValueError, IndexError):
                print("Invalid input. Format: x y z [optional_angle]")
            except KeyboardInterrupt:
                break

if __name__ == "__main__":
    bot = PRRR_Robot()
    bot.run()