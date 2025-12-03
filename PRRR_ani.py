import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
import time
# ============================================================================
# D-H PARAMETERS (Standard DH Convention)
# ============================================================================
# Format: [theta, d, a, alpha]
# 'Var' indicates the variable controlled by the motor
DH_TABLE = [
    {'type': 'P', 'theta': 0.0, 'd': 'Var', 'a': 0.0, 'alpha': 0.0},  # Joint 1 (Lift)
    {'type': 'R', 'theta': 'Var', 'd': 0.0, 'a': 3.5, 'alpha': 0.0},  # Joint 2 (Shoulder)
    {'type': 'R', 'theta': 'Var', 'd': 0.0, 'a': 2.5, 'alpha': 0.0},  # Joint 3 (Elbow)
    {'type': 'R', 'theta': 'Var', 'd': 0.0, 'a': 1.5, 'alpha': 0.0}   # Joint 4 (Wrist)
]
# ============================================================================
# ROBOT PARAMETERS
# ============================================================================
MAX_Z_HEIGHT = 5.0
L_SHOULDER = 3.5
L_ELBOW = 2.5
L_WRIST = 1.5

# Speeds
STEP_SIZE_ANG = 5.0 
STEP_SIZE_LIN = 0.2
SAFE_HOVER_HEIGHT = 2.0 # How high to lift object before moving

class Box:
    def __init__(self, x, y, z):
        self.pos = np.array([x, y, z], dtype=float)
        self.color = 'blue' # Blue = Resting, Red = Gripped

class PRRR_Robot:
    def __init__(self):
        # Robot State
        self.d1 = 0.0
        self.theta1 = 0.0
        self.theta2 = 0.0
        self.theta3 = 0.0
        
        # Gripper State
        self.held_object = None # Reference to the Box object
        
        # Visualization
        self.fig = plt.figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.running = True
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)

    def on_key(self, event):
        if event.key == 'q':
            self.running = False
            plt.close(self.fig)

    # --- KINEMATICS (Same as before) ---
    def forward_kinematics(self, d1, th1, th2, th3):
        p_base = np.array([0, 0, 0])
        p_lift = np.array([0, 0, d1])
        p_shoulder = p_lift
        p_elbow = p_shoulder + np.array([L_SHOULDER * np.cos(th1), L_SHOULDER * np.sin(th1), 0])
        global_angle_2 = th1 + th2
        p_wrist = p_elbow + np.array([L_ELBOW * np.cos(global_angle_2), L_ELBOW * np.sin(global_angle_2), 0])
        global_angle_3 = global_angle_2 + th3
        p_ee = p_wrist + np.array([L_WRIST * np.cos(global_angle_3), L_WRIST * np.sin(global_angle_3), 0])
        return p_base, p_lift, p_elbow, p_wrist, p_ee

    def inverse_kinematics(self, x, y, z, target_phi_deg=None):
        # [Same "Smart Reach" IK logic as previous step]
        if not (0 <= z <= MAX_Z_HEIGHT): raise ValueError(f"Z={z} out of range.")
        d1 = z
        phi_to_try = 0.0 if target_phi_deg is None else target_phi_deg
        max_planar_reach = L_SHOULDER + L_ELBOW

        def solve_arm(tx, ty, phi_rad):
            wx = tx - L_WRIST * np.cos(phi_rad)
            wy = ty - L_WRIST * np.sin(phi_rad)
            r = np.sqrt(wx**2 + wy**2)
            if r > max_planar_reach: return None, None, None, r
            cos_t2 = (r**2 - L_SHOULDER**2 - L_ELBOW**2) / (2 * L_SHOULDER * L_ELBOW)
            t2 = np.arccos(np.clip(cos_t2, -1.0, 1.0))
            t1 = np.arctan2(wy, wx) - np.arctan2(L_ELBOW * np.sin(t2), L_SHOULDER + L_ELBOW * np.cos(t2))
            t3 = phi_rad - (t1 + t2)
            return t1, t2, t3, r

        t1, t2, t3, dist_err = solve_arm(x, y, np.radians(phi_to_try))
        if t1 is not None: return d1, t1, t2, t3
        
        # Smart Reach Fallback
        angle_to_target = np.arctan2(y, x)
        total_dist = np.sqrt(x**2 + y**2)
        if total_dist <= L_SHOULDER + L_ELBOW + L_WRIST:
            print(f"  [Auto-Align] Reaching straight to target.")
            t1, t2, t3, _ = solve_arm(x, y, angle_to_target)
            if t1 is not None: return d1, t1, t2, t3
            
        raise ValueError(f"Target unreachable.")

    # --- MOVEMENT PRIMITIVES ---
    def move_to(self, x, y, z, phi=None, description="Moving"):
        """Low-level function to execute a single straight-line motion"""
        try:
            td1, tt1, tt2, tt3 = self.inverse_kinematics(x, y, z, phi)
            
            # Interpolate
            delta_d1 = td1 - self.d1
            delta_t1, delta_t2, delta_t3 = tt1-self.theta1, tt2-self.theta2, tt3-self.theta3
            
            steps = int(max(abs(delta_d1)/STEP_SIZE_LIN, np.degrees(max(abs(delta_t1), abs(delta_t2), abs(delta_t3)))/STEP_SIZE_ANG))
            if steps < 2: steps = 2
            
            path_d1 = np.linspace(self.d1, td1, steps)
            path_t1 = np.linspace(self.theta1, tt1, steps)
            path_t2 = np.linspace(self.theta2, tt2, steps)
            path_t3 = np.linspace(self.theta3, tt3, steps)
            
            for d, t1, t2, t3 in zip(path_d1, path_t1, path_t2, path_t3):
                if not self.running: break
                
                # Update Robot State
                self.d1, self.theta1, self.theta2, self.theta3 = d, t1, t2, t3
                
                # If holding object, update object position to match End Effector
                if self.held_object:
                    _, _, _, _, p_ee = self.forward_kinematics(d, t1, t2, t3)
                    self.held_object.pos = p_ee # Snap box to gripper
                    
                self.draw_scene([box_obj]) # Pass the box to draw
            
        except ValueError as e:
            print(f"  Move Failed: {e}")
            return False
        return True

    def pick_and_place(self, box, drop_x, drop_y, drop_z):
        """High-level Task Sequencer"""
        print(f"\n--- TASK START: Move Box to ({drop_x}, {drop_y}, {drop_z}) ---")
        
        start_x, start_y, start_z = box.pos
        
        # 1. Approach (Hover above Box)
        print("1. Approaching...")
        if not self.move_to(start_x, start_y, start_z + SAFE_HOVER_HEIGHT): return
        
        # 2. Descend to Box
        print("2. Descending...")
        if not self.move_to(start_x, start_y, start_z): return
        
        # 3. Grip
        print("3. GRIPPING Object")
        self.held_object = box
        box.color = 'red' # Visual feedback
        time.sleep(0.5) # Simulate grip time
        
        # 4. Lift (Retract)
        print("4. Lifting...")
        if not self.move_to(start_x, start_y, start_z + SAFE_HOVER_HEIGHT): return
        
        # 5. Traverse (Hover above Drop Zone)
        print("5. Traversing to Drop Zone...")
        if not self.move_to(drop_x, drop_y, drop_z + SAFE_HOVER_HEIGHT): return
        
        # 6. Descend to Drop
        print("6. Placing...")
        if not self.move_to(drop_x, drop_y, drop_z): return
        
        # 7. Release
        print("7. RELEASING Object")
        self.held_object = None
        box.color = 'blue'
        time.sleep(0.5)
        
        # 8. Home/Retract
        print("8. Retracting...")
        self.move_to(drop_x, drop_y, drop_z + SAFE_HOVER_HEIGHT)
        print("--- TASK COMPLETE ---")

    def draw_scene(self, world_objects):
        self.ax.cla()
        self.ax.set_xlim(-8, 8); self.ax.set_ylim(-8, 8); self.ax.set_zlim(0, 6)
        self.ax.set_title("Pick & Place Simulation\nBlue=Resting, Red=Gripped")
        
        # Draw Robot
        base, lift, elbow, wrist, ee = self.forward_kinematics(self.d1, self.theta1, self.theta2, self.theta3)
        
        # Links
        self.ax.plot([0,0], [0,0], [0, MAX_Z_HEIGHT], 'k--', alpha=0.3) # Rail
        self.ax.plot([base[0], lift[0]], [base[1], lift[1]], [base[2], lift[2]], 'k-', lw=6)
        self.ax.plot([lift[0], elbow[0]], [lift[1], elbow[1]], [lift[2], elbow[2]], 'b-', lw=4)
        self.ax.plot([elbow[0], wrist[0]], [elbow[1], wrist[1]], [elbow[2], wrist[2]], 'g-', lw=4)
        self.ax.plot([wrist[0], ee[0]], [wrist[1], ee[1]], [wrist[2], ee[2]], 'r-', lw=2)
        
        # Joints
        self.ax.scatter(*lift, s=100, c='k', marker='s')
        self.ax.scatter(*elbow, s=80, c='b')
        self.ax.scatter(*wrist, s=80, c='g')
        
        # Draw Objects
        for obj in world_objects:
            self.ax.scatter(obj.pos[0], obj.pos[1], obj.pos[2], 
                           s=200, marker='s', c=obj.color, edgecolors='k', alpha=0.9)

        plt.draw()
        plt.pause(0.001)

if __name__ == "__main__":
    # Setup
    sim = PRRR_Robot()
    
    # Create a Box at a random reachable location (fixed Z=0 for floor pick)
    box_start_x = random.uniform(3, 6)
    box_start_y = random.uniform(-3, 3)
    box_obj = Box(box_start_x, box_start_y, 0.0)
    
    # Initial Draw
    sim.draw_scene([box_obj])
    plt.show(block=False)
    
    print(f"WORLD STATE: Box located at ({box_obj.pos[0]:.2f}, {box_obj.pos[1]:.2f}, 0.0)")

    while sim.running:
        try:
            raw = input("\nEnter Drop-off Coordinates (x y z) or 'q': ")
            if raw.lower() == 'q': break
            
            coords = [float(v) for v in raw.replace(',', ' ').split()]
            if len(coords) < 3: 
                print("Need x, y, z")
                continue
                
            # Execute Full Sequence
            sim.pick_and_place(box_obj, coords[0], coords[1], coords[2])
            
        except ValueError:
            print("Invalid Input.")
        except KeyboardInterrupt:
            break