import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import TextBox, CheckButtons
import random
import time

# ============================================================================
# D-H PARAMETERS (Standard DH Convention)
# ============================================================================
DH_TABLE = [
    {'type': 'P', 'theta': 0.0, 'd': 'Var', 'a': 0.0, 'alpha': 0.0},  # J1: Lift
    {'type': 'R', 'theta': 'Var', 'd': 0.0, 'a': 3.5, 'alpha': 0.0},  # J2: Shoulder
    {'type': 'R', 'theta': 'Var', 'd': 0.0, 'a': 2.5, 'alpha': 0.0},  # J3: Elbow
    {'type': 'R', 'theta': 'Var', 'd': 0.0, 'a': 1.5, 'alpha': 0.0}   # J4: Wrist
]

# ============================================================================
# ROBOT CONSTANTS
# ============================================================================
MAX_Z_HEIGHT = 5.0
L_SHOULDER = 3.5
L_ELBOW = 2.5
L_WRIST = 1.5
MAX_REACH = L_SHOULDER + L_ELBOW + L_WRIST  # 7.5

STEP_SIZE_ANG = 5.0 
STEP_SIZE_LIN = 0.2
SAFE_HOVER_HEIGHT = 2.0 

class Box:
    def __init__(self, x, y, z):
        self.pos = np.array([x, y, z], dtype=float)
        self.color = 'blue'

class PRRR_Robot:
    def __init__(self):
        self.d1 = 0.0
        self.theta1 = 0.0
        self.theta2 = 0.0
        self.theta3 = 0.0
        self.held_object = None
        self.show_workspace = False
        
        # Setup Figure with extra space at bottom for controls
        self.fig = plt.figure(figsize=(10, 9))
        self.fig.subplots_adjust(bottom=0.2) # Make room for GUI
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # --- GUI ELEMENTS ---
        # 1. Workspace Checkbox
        ax_check = plt.axes([0.05, 0.05, 0.2, 0.1]) # Position: Left Bottom
        self.chk_box = CheckButtons(ax_check, ['Show Workspace'], [False])
        self.chk_box.on_clicked(self.toggle_workspace)
        
        # 2. Input Textbox
        ax_box = plt.axes([0.35, 0.05, 0.4, 0.075]) # Position: Center Bottom
        self.text_box = TextBox(ax_box, 'Target (x y z): ', initial="4 0 0")
        self.text_box.on_submit(self.on_submit)

        # Pre-calculate Workspace Mesh (Optimization)
        self.ws_mesh = self.generate_workspace_data()

    def generate_workspace_data(self):
        """Generates the cylinder mesh for the workspace overlay."""
        # Cylinder Parameters
        radius = MAX_REACH
        height = MAX_Z_HEIGHT
        resolution = 50
        
        # Generate Grid
        theta = np.linspace(0, 2*np.pi, resolution)
        z = np.linspace(0, height, resolution)
        theta_grid, z_grid = np.meshgrid(theta, z)
        
        x_grid = radius * np.cos(theta_grid)
        y_grid = radius * np.sin(theta_grid)
        
        return x_grid, y_grid, z_grid

    def toggle_workspace(self, label):
        self.show_workspace = not self.show_workspace
        self.draw_scene([box_obj] if 'box_obj' in globals() else [])

    def on_submit(self, text):
        """Callback when user hits Enter in the text box"""
        try:
            coords = [float(v) for v in text.replace(',', ' ').split()]
            if len(coords) < 3: return
            
            # Run the sequence
            self.pick_and_place(box_obj, coords[0], coords[1], coords[2])
            
        except ValueError:
            print("Invalid Input")

    # --- KINEMATICS & IK (Same logic) ---
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
        if total_dist <= MAX_REACH:
            print(f"  [Smart Reach] Auto-Aligning to Target...")
            t1, t2, t3, _ = solve_arm(x, y, angle_to_target)
            if t1 is not None: return d1, t1, t2, t3
            
        raise ValueError(f"Target Unreachable.")

    def move_to(self, x, y, z, phi=None):
        try:
            td1, tt1, tt2, tt3 = self.inverse_kinematics(x, y, z, phi)
            
            delta_d1 = td1 - self.d1
            delta_t1, delta_t2, delta_t3 = tt1-self.theta1, tt2-self.theta2, tt3-self.theta3
            
            steps = int(max(abs(delta_d1)/STEP_SIZE_LIN, np.degrees(max(abs(delta_t1), abs(delta_t2), abs(delta_t3)))/STEP_SIZE_ANG))
            if steps < 2: steps = 2
            
            path_d1 = np.linspace(self.d1, td1, steps)
            path_t1 = np.linspace(self.theta1, tt1, steps)
            path_t2 = np.linspace(self.theta2, tt2, steps)
            path_t3 = np.linspace(self.theta3, tt3, steps)
            
            for d, t1, t2, t3 in zip(path_d1, path_t1, path_t2, path_t3):
                self.d1, self.theta1, self.theta2, self.theta3 = d, t1, t2, t3
                if self.held_object:
                    _, _, _, _, p_ee = self.forward_kinematics(d, t1, t2, t3)
                    self.held_object.pos = p_ee 
                self.draw_scene([box_obj])
            return True
        except ValueError as e:
            print(f"  Move Failed: {e}")
            return False

    def pick_and_place(self, box, drop_x, drop_y, drop_z):
        print(f"\nTask: Move Box to ({drop_x}, {drop_y}, {drop_z})")
        
        # 1. READ STATE
        start_x, start_y, start_z = box.pos
        
        # 2. CALCULATE SAFE HEIGHTS (THE FIX)
        # We assume the safest travel height is the highest we can go (Ceiling),
        # but we clamp it so we never ask for Z > 5.0.
        
        # Lift Height: Start Z + 2.0, but capped at 5.0
        lift_z = min(start_z + SAFE_HOVER_HEIGHT, MAX_Z_HEIGHT)
        
        # Drop Hover Height: Target Z + 2.0, but capped at 5.0
        drop_hover_z = min(drop_z + SAFE_HOVER_HEIGHT, MAX_Z_HEIGHT)
        
        # Traverse Height: The higher of the two, capped at 5.0
        traverse_z = max(lift_z, drop_hover_z)

        # 3. EXECUTE SEQUENCE
        # A. Approach & Descend (Pick)
        print(f"  > Picking from Z={start_z}...")
        if not self.move_to(start_x, start_y, lift_z): return       # Hover above start
        if not self.move_to(start_x, start_y, start_z): return      # Descend
        
        # B. Grip
        self.held_object = box
        box.color = 'red'
        time.sleep(0.2)
        
        # C. Lift & Traverse
        print(f"  > Moving to Z={drop_z}...")
        if not self.move_to(start_x, start_y, lift_z): return       # Retract Z
        if not self.move_to(drop_x, drop_y, traverse_z): return     # Fly to Destination (XY)
        
        # D. Descend & Release (Place)
        # Note: We move to drop_hover_z first to ensure we come down vertically
        if not self.move_to(drop_x, drop_y, drop_hover_z): return   # Align Z
        if not self.move_to(drop_x, drop_y, drop_z): return         # Place
        
        self.held_object = None
        box.color = 'blue'
        time.sleep(0.2)
        
        # E. Retract
        self.move_to(drop_x, drop_y, drop_hover_z)
        print("Done.")

    def draw_scene(self, world_objects):
        self.ax.cla()
        self.ax.set_xlim(-8, 8); self.ax.set_ylim(-8, 8); self.ax.set_zlim(0, 6)
        self.ax.set_xlabel('X'); self.ax.set_ylabel('Y'); self.ax.set_zlabel('Z')
        self.ax.set_title("PRRR Robot Control Center")
        
        # --- 1. Draw Workspace Overlay (If Checked) ---
        if self.show_workspace:
            X, Y, Z = self.ws_mesh
            # alpha=0.1 gives the "minimal opacity" ghost effect
            self.ax.plot_surface(X, Y, Z, color='yellow', alpha=0.1, rstride=5, cstride=5)
            # Add wireframe edges for definition
            # self.ax.plot_wireframe(X, Y, Z, color='orange', alpha=0.2, rstride=10, cstride=10)

        # --- 2. Draw Robot ---
        base, lift, elbow, wrist, ee = self.forward_kinematics(self.d1, self.theta1, self.theta2, self.theta3)
        self.ax.plot([0,0], [0,0], [0, MAX_Z_HEIGHT], 'k--', alpha=0.3)
        self.ax.plot([base[0], lift[0]], [base[1], lift[1]], [base[2], lift[2]], 'k-', lw=6)
        self.ax.plot([lift[0], elbow[0]], [lift[1], elbow[1]], [lift[2], elbow[2]], 'b-', lw=4)
        self.ax.plot([elbow[0], wrist[0]], [elbow[1], wrist[1]], [elbow[2], wrist[2]], 'g-', lw=4)
        self.ax.plot([wrist[0], ee[0]], [wrist[1], ee[1]], [wrist[2], ee[2]], 'r-', lw=2)
        
        self.ax.scatter(*lift, s=100, c='k', marker='s')
        self.ax.scatter(*elbow, s=80, c='b')
        self.ax.scatter(*wrist, s=80, c='g')
        
        # --- 3. Draw Objects ---
        for obj in world_objects:
            self.ax.scatter(obj.pos[0], obj.pos[1], obj.pos[2], 
                           s=200, marker='s', c=obj.color, edgecolors='k', alpha=0.9)

        plt.draw()
        plt.pause(0.001)

if __name__ == "__main__":
    box_start_x = random.uniform(3, 6)
    box_start_y = random.uniform(-3, 3)
    box_obj = Box(box_start_x, box_start_y, 0.0)
    
    sim = PRRR_Robot()
    sim.draw_scene([box_obj])
    
    print("Use the GUI Control Panel on the plot window.")
    plt.show() # BLOCKING - Controls work here