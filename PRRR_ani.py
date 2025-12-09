import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import TextBox, CheckButtons
import random
import time

# ============================================================================
# D-H PARAMETERS
# ============================================================================
DH_TABLE = [
    {'type': 'P', 'theta': 0.0, 'd': 'Var', 'a': 0.0, 'alpha': 0.0},
    {'type': 'R', 'theta': 'Var', 'd': 0.0, 'a': 3.5, 'alpha': 0.0},
    {'type': 'R', 'theta': 'Var', 'd': 0.0, 'a': 2.5, 'alpha': 0.0},
    {'type': 'R', 'theta': 'Var', 'd': 0.0, 'a': 1.5, 'alpha': 0.0}
]

# ============================================================================
# ROBOT CONSTANTS
# ============================================================================
MAX_Z_HEIGHT = 5.0
L_SHOULDER = 3.5
L_ELBOW = 2.5
L_WRIST = 1.5
MAX_REACH = L_SHOULDER + L_ELBOW + L_WRIST

STEP_SIZE_ANG = 5.0 
STEP_SIZE_LIN = 0.2
SAFE_HOVER_HEIGHT = 2.0 

# ============================================================================
# CLASSES
# ============================================================================
class WorldObject:
    def __init__(self, name, color, marker, start_pos, dest_pos=None):
        self.name = name
        self.color = color
        self.marker = marker # 's'=square, '^'=triangle, 'o'=circle
        self.pos = np.array(start_pos, dtype=float)
        self.dest = np.array(dest_pos, dtype=float) if dest_pos else None
        self.home_pos = np.array(start_pos, dtype=float) # To remember where it started

class PRRR_Robot:
    def __init__(self):
        self.d1 = 0.0
        self.theta1 = 0.0
        self.theta2 = 0.0
        self.theta3 = 0.0
        self.held_object = None
        self.show_workspace = False
        
        # --- INITIALIZE OBJECTS ---
        # 1. Blue Square (Random Start, User Input Dest)
        sq_pos = [random.uniform(3, 6), random.uniform(-3, 3), 0.0]
        self.blue_square = WorldObject('Blue Square', 'blue', 's', sq_pos)

        # 2. Green Triangle (Random Start, Random Dest at Z=2)
        tri_start = [random.uniform(3, 6), random.uniform(-3, 3), 0.0]
        tri_dest = [random.uniform(-4, 4), random.uniform(-4, 4), 2.0]
        self.green_triangle = WorldObject('Green Triangle', 'green', '^', tri_start, tri_dest)

        # 3. Yellow Circle (Random Start, Random Dest at Z=4)
        circ_start = [random.uniform(3, 6), random.uniform(-3, 3), 0.0]
        circ_dest = [random.uniform(-4, 4), random.uniform(-4, 4), 4.0]
        self.yellow_circle = WorldObject('Yellow Circle', 'yellow', 'o', circ_start, circ_dest)
        
        # List for iteration
        self.world_objects = [self.blue_square, self.green_triangle, self.yellow_circle]

        # --- GUI SETUP ---
        self.fig = plt.figure(figsize=(12, 9))
        self.fig.subplots_adjust(left=0.05, bottom=0.25) # More room for controls
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # 1. Checkboxes (Workspace + Start Buttons)
        ax_check = plt.axes([0.05, 0.05, 0.25, 0.15]) 
        self.labels = ['Show Workspace', 'Start Triangle', 'Start Circle']
        self.chk_box = CheckButtons(ax_check, self.labels, [False, False, False])
        self.chk_box.on_clicked(self.on_checkbox_click)
        
        # 2. Textbox (For Blue Square)
        ax_box = plt.axes([0.4, 0.05, 0.3, 0.05])
        self.text_box = TextBox(ax_box, 'Blue Square Dest (x y z): ', initial="0 5 0")
        self.text_box.on_submit(self.on_submit_square)

        # Pre-calc Workspace
        self.ws_mesh = self.generate_workspace_data()

    def generate_workspace_data(self):
        radius = MAX_REACH
        height = MAX_Z_HEIGHT
        theta = np.linspace(0, 2*np.pi, 30)
        z = np.linspace(0, height, 10)
        theta_grid, z_grid = np.meshgrid(theta, z)
        x_grid = radius * np.cos(theta_grid)
        y_grid = radius * np.sin(theta_grid)
        return x_grid, y_grid, z_grid

    # --- CALLBACKS ---
    def on_checkbox_click(self, label):
        if label == 'Show Workspace':
            self.show_workspace = not self.show_workspace
            self.draw_scene()
        elif label == 'Start Triangle':
            # Run task for Green Triangle
            self.pick_and_place(self.green_triangle, *self.green_triangle.dest)
        elif label == 'Start Circle':
            # Run task for Yellow Circle
            self.pick_and_place(self.yellow_circle, *self.yellow_circle.dest)

    def on_submit_square(self, text):
        try:
            coords = [float(v) for v in text.replace(',', ' ').split()]
            if len(coords) < 3: return
            self.pick_and_place(self.blue_square, coords[0], coords[1], coords[2])
        except ValueError:
            print("Invalid Input")

    # --- KINEMATICS ---
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
            print(f"  [Smart Reach] Auto-Aligning...")
            t1, t2, t3, _ = solve_arm(x, y, angle_to_target)
            if t1 is not None: return d1, t1, t2, t3
            
        raise ValueError(f"Target Unreachable.")

    # --- MOTION ---
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
                self.draw_scene()
            return True
        except ValueError as e:
            print(f"  Move Failed: {e}")
            return False

    def pick_and_place(self, obj, drop_x, drop_y, drop_z):
        print(f"\nTask: Move {obj.name} to ({drop_x:.1f}, {drop_y:.1f}, {drop_z:.1f})")
        start_x, start_y, start_z = obj.pos
        
        # Z-Safety Logic
        lift_z = min(start_z + SAFE_HOVER_HEIGHT, MAX_Z_HEIGHT)
        drop_hover_z = min(drop_z + SAFE_HOVER_HEIGHT, MAX_Z_HEIGHT)
        traverse_z = max(lift_z, drop_hover_z)

        # Sequence
        if not self.move_to(start_x, start_y, lift_z): return
        if not self.move_to(start_x, start_y, start_z): return
        
        self.held_object = obj; obj.color = 'red'; time.sleep(0.2) # GRIP
        
        if not self.move_to(start_x, start_y, lift_z): return
        if not self.move_to(drop_x, drop_y, traverse_z): return
        if not self.move_to(drop_x, drop_y, drop_hover_z): return
        if not self.move_to(drop_x, drop_y, drop_z): return
        
        self.held_object = None; obj.color = 'blue' if 'Square' in obj.name else ('green' if 'Triangle' in obj.name else 'yellow'); 
        time.sleep(0.2) # RELEASE
        
        self.move_to(drop_x, drop_y, drop_hover_z)
        print("Done.")

    def draw_scene(self):
        self.ax.cla()
        self.ax.set_xlim(-8, 8); self.ax.set_ylim(-8, 8); self.ax.set_zlim(0, 6)
        self.ax.set_title("PRRR Robot Control Center")
        
        # 1. Workspace
        if self.show_workspace:
            X, Y, Z = self.ws_mesh
            self.ax.plot_surface(X, Y, Z, color='yellow', alpha=0.1)

        # 2. Robot
        base, lift, elbow, wrist, ee = self.forward_kinematics(self.d1, self.theta1, self.theta2, self.theta3)
        self.ax.plot([0,0], [0,0], [0, MAX_Z_HEIGHT], 'k--', alpha=0.3)
        self.ax.plot([base[0], lift[0]], [base[1], lift[1]], [base[2], lift[2]], 'k-', lw=6)
        self.ax.plot([lift[0], elbow[0]], [lift[1], elbow[1]], [lift[2], elbow[2]], 'b-', lw=4)
        self.ax.plot([elbow[0], wrist[0]], [elbow[1], wrist[1]], [elbow[2], wrist[2]], 'g-', lw=4)
        self.ax.plot([wrist[0], ee[0]], [wrist[1], ee[1]], [wrist[2], ee[2]], 'r-', lw=2)
        
        # 3. Objects & Destinations
        for obj in self.world_objects:
            # Draw Object
            self.ax.scatter(obj.pos[0], obj.pos[1], obj.pos[2], 
                           s=150, marker=obj.marker, c=obj.color, edgecolors='k', alpha=1.0)
            
            # Draw Destination Marker (if it has a fixed dest)
            if obj.dest is not None:
                dx, dy, dz = obj.dest
                self.ax.scatter(dx, dy, dz, c='red', marker='x', s=50)
                label_name = "tri_shelf" if 'Triangle' in obj.name else "circ_shelf"
                self.ax.text(dx, dy, dz, f" {label_name}", color='red', fontsize=8)

        plt.draw()
        plt.pause(0.001)

if __name__ == "__main__":
    sim = PRRR_Robot()
    sim.draw_scene()
    print("GUI Ready.")
    plt.show()