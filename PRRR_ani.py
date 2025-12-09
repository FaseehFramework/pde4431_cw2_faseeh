import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import TextBox, CheckButtons, Button
import random
import time

# ============================================================================
# CONFIGURATION & PARAMETERS
# ============================================================================
# D-H Table (Standard)
DH_TABLE = [
    {'type': 'P', 'theta': 0.0, 'd': 'Var', 'a': 0.0, 'alpha': 0.0},
    {'type': 'R', 'theta': 'Var', 'd': 0.0, 'a': 3.5, 'alpha': 0.0},
    {'type': 'R', 'theta': 'Var', 'd': 0.0, 'a': 2.5, 'alpha': 0.0},
    {'type': 'R', 'theta': 'Var', 'd': 0.0, 'a': 1.5, 'alpha': 0.0}
]

# Robot Geometry
MAX_Z_HEIGHT = 5.0
L_SHOULDER, L_ELBOW, L_WRIST = 3.5, 2.5, 1.5
MAX_REACH = L_SHOULDER + L_ELBOW + L_WRIST

# Simulation Settings
STEP_SIZE_ANG = 5.0 
STEP_SIZE_LIN = 0.2
SAFE_HOVER_HEIGHT = 2.0
OBJECT_HEIGHT = 0.6     # Physical height of blocks
STACK_TOLERANCE = 0.5   # Distance to trigger stacking

# ============================================================================
# CLASSES
# ============================================================================
class WorldObject:
    def __init__(self, name, color, marker, start_pos, dest_pos=None):
        self.name = name
        self.color = color
        self.marker = marker 
        self.pos = np.array(start_pos, dtype=float)
        self.dest = np.array(dest_pos, dtype=float) if dest_pos else None
        self.height = OBJECT_HEIGHT

class RobotController:
    def __init__(self):
        # 1. Initialize Robot State
        self.d1, self.theta1, self.theta2, self.theta3 = 0.0, 0.0, 0.0, 0.0
        self.held_object = None
        self.show_workspace = False
        
        # 2. Initialize World Objects (Randomized)
        self.init_world()

        # 3. Setup Visualization Window (Window 1)
        self.fig_viz = plt.figure(figsize=(8, 8))
        self.fig_viz.canvas.manager.set_window_title('Visualization')
        self.ax = self.fig_viz.add_subplot(111, projection='3d')
        
        # 4. Setup Control Panel Window (Window 2)
        self.setup_control_panel()

        # Pre-calc Workspace Data
        self.ws_mesh = self.generate_workspace_data()
        
        # Initial Draw
        self.draw_scene()

    def init_world(self):
        # Blue Square (User Input)
        sq_pos = [random.uniform(3, 6), random.uniform(-3, 3), 0.0]
        self.blue_square = WorldObject('Blue Square', 'blue', 's', sq_pos)

        # Green Triangle (Random Dest Z=2)
        tri_start = [random.uniform(3, 6), random.uniform(-3, 3), 0.0]
        tri_dest = [random.uniform(-4, 4), random.uniform(-4, 4), 2.0]
        self.green_triangle = WorldObject('Green Triangle', 'green', '^', tri_start, tri_dest)

        # Yellow Circle (Random Dest Z=4)
        circ_start = [random.uniform(3, 6), random.uniform(-3, 3), 0.0]
        circ_dest = [random.uniform(-4, 4), random.uniform(-4, 4), 4.0]
        self.yellow_circle = WorldObject('Yellow Circle', 'yellow', 'o', circ_start, circ_dest)
        
        self.world_objects = [self.blue_square, self.green_triangle, self.yellow_circle]

    def setup_control_panel(self):
        self.fig_ctrl = plt.figure(figsize=(5, 6))
        self.fig_ctrl.canvas.manager.set_window_title('Command Center')
        
        # Title
        plt.text(0.5, 0.95, "ROBOT CONTROLS", ha='center', fontsize=14, weight='bold')
        plt.axis('off')

        # Section 1: Visuals
        ax_chk = plt.axes([0.1, 0.8, 0.4, 0.1])
        self.chk_ws = CheckButtons(ax_chk, ['Show Workspace'], [False])
        self.chk_ws.on_clicked(self.toggle_workspace)

        # Section 2: Automated Tasks
        plt.text(0.1, 0.75, "Auto-Tasks (Hardcoded)", fontsize=10, weight='bold', transform=self.fig_ctrl.transFigure)
        
        ax_btn_tri = plt.axes([0.1, 0.65, 0.35, 0.08])
        self.btn_tri = Button(ax_btn_tri, 'Move Triangle', color='lightgreen', hovercolor='0.9')
        self.btn_tri.on_clicked(lambda x: self.run_task(self.green_triangle))
        
        ax_btn_circ = plt.axes([0.55, 0.65, 0.35, 0.08])
        self.btn_circ = Button(ax_btn_circ, 'Move Circle', color='khaki', hovercolor='0.9')
        self.btn_circ.on_clicked(lambda x: self.run_task(self.yellow_circle))

        # Section 3: User Task
        plt.text(0.1, 0.55, "User Task (Blue Square)", fontsize=10, weight='bold', transform=self.fig_ctrl.transFigure)
        
        ax_txt = plt.axes([0.1, 0.48, 0.5, 0.05])
        self.txt_input = TextBox(ax_txt, 'Dest (x y z): ', initial="0 5 0")
        
        ax_btn_sq = plt.axes([0.65, 0.48, 0.25, 0.05])
        self.btn_sq = Button(ax_btn_sq, 'GO', color='lightblue')
        self.btn_sq.on_clicked(self.on_square_submit)

        # Section 4: Global Reset
        ax_btn_rst = plt.axes([0.1, 0.1, 0.8, 0.1])
        self.btn_rst = Button(ax_btn_rst, 'RESET SIMULATION', color='salmon')
        self.btn_rst.on_clicked(self.reset_sim)

    # --- CORE LOGIC ---
    def get_smart_z(self, x, y, moving_obj_name):
        """Scans (x,y) for existing objects to stack on."""
        max_h = 0.0
        for obj in self.world_objects:
            if obj.name == moving_obj_name: continue # Don't check self
            
            # Check 2D distance
            dist = np.sqrt((obj.pos[0]-x)**2 + (obj.pos[1]-y)**2)
            if dist < STACK_TOLERANCE:
                top = obj.pos[2] + obj.height
                if top > max_h: max_h = top
                print(f"  [Smart Stack] Detected {obj.name} base at Z={obj.pos[2]:.1f}")
        return max_h

    def run_task(self, obj):
        if obj.dest is None: return
        # Use Smart Stacking on the hardcoded destination
        base_z = self.get_smart_z(obj.dest[0], obj.dest[1], obj.name)
        final_z = base_z + obj.dest[2] # Add Hardcoded Offset to the stack base
        
        # Override the destination Z with the smart calculation
        # Note: For Triangle/Circle, the requirement was "Z is at level 2/4".
        # If we stack, should we add 2 to the stack? Or be at absolute 2?
        # Logic: We preserve the absolute Z requirement, but lift if stack is higher.
        target_z = max(obj.dest[2], base_z) 
        
        self.pick_and_place(obj, obj.dest[0], obj.dest[1], target_z)

    def on_square_submit(self, event):
        try:
            txt = self.txt_input.text
            v = [float(x) for x in txt.replace(',', ' ').split()]
            if len(v) < 3: return
            
            # Smart Logic for User Input
            base_z = self.get_smart_z(v[0], v[1], self.blue_square.name)
            target_z = base_z + v[2] # User's Z is treated as offset from whatever is there
            
            self.pick_and_place(self.blue_square, v[0], v[1], target_z)
        except ValueError:
            print("Invalid Input")

    def toggle_workspace(self, event):
        self.show_workspace = not self.show_workspace
        self.draw_scene()

    def reset_sim(self, event):
        print("Resetting...")
        self.init_world()
        self.d1, self.theta1, self.theta2, self.theta3 = 0,0,0,0
        self.held_object = None
        self.draw_scene()

    # --- KINEMATICS & MOTION ---
    def generate_workspace_data(self):
        theta = np.linspace(0, 2*np.pi, 30); z = np.linspace(0, MAX_Z_HEIGHT, 10)
        T, Z = np.meshgrid(theta, z)
        X = MAX_REACH * np.cos(T); Y = MAX_REACH * np.sin(T)
        return X, Y, Z

    def forward_kinematics(self, d1, th1, th2, th3):
        p0 = np.array([0, 0, 0])
        p1 = np.array([0, 0, d1])
        p2 = p1 + np.array([L_SHOULDER*np.cos(th1), L_SHOULDER*np.sin(th1), 0])
        g2 = th1 + th2
        p3 = p2 + np.array([L_ELBOW*np.cos(g2), L_ELBOW*np.sin(g2), 0])
        g3 = g2 + th3
        p4 = p3 + np.array([L_WRIST*np.cos(g3), L_WRIST*np.sin(g3), 0])
        return p0, p1, p2, p3, p4

    def inverse_kinematics(self, x, y, z):
        if not (0 <= z <= MAX_Z_HEIGHT): raise ValueError(f"Z={z} out of range.")
        d1 = z; max_reach = L_SHOULDER + L_ELBOW
        
        def solve(tx, ty, phi):
            wx, wy = tx - L_WRIST*np.cos(phi), ty - L_WRIST*np.sin(phi)
            r = np.sqrt(wx**2 + wy**2)
            if r > max_reach: return None, None, None
            c2 = (r**2 - L_SHOULDER**2 - L_ELBOW**2)/(2*L_SHOULDER*L_ELBOW)
            t2 = np.arccos(np.clip(c2, -1, 1))
            t1 = np.arctan2(wy, wx) - np.arctan2(L_ELBOW*np.sin(t2), L_SHOULDER + L_ELBOW*np.cos(t2))
            t3 = phi - (t1+t2)
            return t1, t2, t3

        # 1. Try Orient 0
        t1, t2, t3 = solve(x, y, 0)
        if t1 is not None: return d1, t1, t2, t3
        
        # 2. Try Point at Target
        angle = np.arctan2(y, x)
        if np.sqrt(x**2 + y**2) <= MAX_REACH:
            t1, t2, t3 = solve(x, y, angle)
            if t1 is not None: return d1, t1, t2, t3
            
        raise ValueError("Unreachable")

    def move_to(self, x, y, z):
        try:
            td1, tt1, tt2, tt3 = self.inverse_kinematics(x, y, z)
            delta_d = td1 - self.d1
            max_ang_diff = max(abs(tt1-self.theta1), abs(tt2-self.theta2), abs(tt3-self.theta3))
            steps = int(max(abs(delta_d)/STEP_SIZE_LIN, np.degrees(max_ang_diff)/STEP_SIZE_ANG))
            if steps < 2: steps = 2
            
            # Linear interpolation
            traj_d = np.linspace(self.d1, td1, steps)
            traj_t1 = np.linspace(self.theta1, tt1, steps)
            traj_t2 = np.linspace(self.theta2, tt2, steps)
            traj_t3 = np.linspace(self.theta3, tt3, steps)
            
            for d, t1, t2, t3 in zip(traj_d, traj_t1, traj_t2, traj_t3):
                self.d1, self.theta1, self.theta2, self.theta3 = d, t1, t2, t3
                if self.held_object:
                    _, _, _, _, pee = self.forward_kinematics(d, t1, t2, t3)
                    self.held_object.pos = pee 
                self.draw_scene()
            return True
        except ValueError:
            print("Move failed")
            return False

    def pick_and_place(self, obj, dx, dy, dz):
        sx, sy, sz = obj.pos
        # Calculated Heights
        lift_z = min(sz + SAFE_HOVER_HEIGHT, MAX_Z_HEIGHT)
        drop_hover_z = min(dz + SAFE_HOVER_HEIGHT, MAX_Z_HEIGHT)
        travel_z = max(lift_z, drop_hover_z)
        
        # Sequence
        if not self.move_to(sx, sy, lift_z): return
        if not self.move_to(sx, sy, sz): return
        self.held_object = obj; obj.color = 'red'; time.sleep(0.1)
        if not self.move_to(sx, sy, lift_z): return
        if not self.move_to(dx, dy, travel_z): return
        if not self.move_to(dx, dy, drop_hover_z): return
        if not self.move_to(dx, dy, dz): return
        self.held_object = None; 
        # Restore Colors
        obj.color = 'blue' if 'Square' in obj.name else ('green' if 'Triangle' in obj.name else 'yellow')
        time.sleep(0.1)
        self.move_to(dx, dy, drop_hover_z)

    def draw_scene(self):
        plt.figure(self.fig_viz.number)
        self.ax.cla()
        self.ax.set_xlim(-8, 8); self.ax.set_ylim(-8, 8); self.ax.set_zlim(0, 6)
        self.ax.set_xlabel('X'); self.ax.set_ylabel('Y'); self.ax.set_zlabel('Z')
        self.ax.set_title("Robot Visualization")
        
        # Workspace Overlay
        if self.show_workspace:
            X, Y, Z = self.ws_mesh
            self.ax.plot_surface(X, Y, Z, color='yellow', alpha=0.1)

        # Robot Links
        p0, p1, p2, p3, p4 = self.forward_kinematics(self.d1, self.theta1, self.theta2, self.theta3)
        self.ax.plot([0,0], [0,0], [0, MAX_Z_HEIGHT], 'k--', alpha=0.3)
        self.ax.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]], 'k-', lw=6)
        self.ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 'b-', lw=4)
        self.ax.plot([p2[0], p3[0]], [p2[1], p3[1]], [p2[2], p3[2]], 'g-', lw=4)
        self.ax.plot([p3[0], p4[0]], [p3[1], p4[1]], [p3[2], p4[2]], 'r-', lw=2)
        
        # Robot Joints
        self.ax.scatter(*p1, s=100, c='k', marker='s')
        self.ax.scatter(*p2, s=80, c='b')
        self.ax.scatter(*p3, s=80, c='g')

        # Objects & Targets
        for obj in self.world_objects:
            self.ax.scatter(*obj.pos, s=200, marker=obj.marker, c=obj.color, edgecolors='k', alpha=1.0)
            if obj.dest is not None:
                self.ax.scatter(*obj.dest, c='red', marker='x', s=40)
                lbl = "Tri Shelf" if 'Triangle' in obj.name else "Circ Shelf"
                self.ax.text(obj.dest[0], obj.dest[1], obj.dest[2], lbl, fontsize=8, color='red')

        self.fig_viz.canvas.draw()
        self.fig_viz.canvas.flush_events()

if __name__ == "__main__":
    app = RobotController()
    print("Simulation Started.")
    plt.show()