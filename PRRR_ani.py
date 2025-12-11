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

# Workspace Boundaries
WORKSPACE_X_MIN, WORKSPACE_X_MAX = -7.0, 7.0
WORKSPACE_Y_MIN, WORKSPACE_Y_MAX = -7.0, 7.0
WORKSPACE_Z_MIN, WORKSPACE_Z_MAX = 0.0, MAX_Z_HEIGHT

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
        
        # 2. Initialize World Objects
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
        sq_pos = [random.uniform(0, 6), random.uniform(-3, 3), 0.0]
        self.blue_square = WorldObject('Blue Square', 'blue', 's', sq_pos)

        # Green Triangle (Random Dest Z=2)
        tri_start = [random.uniform(0, 6), random.uniform(-3, 3), 0.0]
        tri_dest = [random.uniform(-5, 5), random.uniform(-5, 5), 2.0]
        self.green_triangle = WorldObject('Green Triangle', 'green', '^', tri_start, tri_dest)

        # Yellow Circle (Random Dest Z=4)
        circ_start = [random.uniform(0, 6), random.uniform(-3, 3), 0.0]
        circ_dest = [random.uniform(-5, 5), random.uniform(-5, 5), 4.0]
        self.yellow_circle = WorldObject('Yellow Circle', 'yellow', 'o', circ_start, circ_dest)
        
        self.world_objects = [self.blue_square, self.green_triangle, self.yellow_circle]
        
        print(f"Green Triangle Dest: {tri_dest}")
        print(f"Yellow Circle Dest: {circ_dest}")

    def setup_control_panel(self):
        self.fig_ctrl = plt.figure(figsize=(5, 7)) # Increased height
        self.fig_ctrl.canvas.manager.set_window_title('Command Center')
        
        plt.text(0.5, 0.95, "ROBOT CONTROLS", ha='center', fontsize=14, weight='bold')
        plt.axis('off')

        # --- Section 1: Visuals ---
        ax_chk = plt.axes([0.1, 0.82, 0.4, 0.08])
        self.chk_ws = CheckButtons(ax_chk, ['Show Workspace'], [False])
        self.chk_ws.on_clicked(self.toggle_workspace)

        # --- Section 2: Auto-Tasks (Individual) ---
        plt.text(0.1, 0.77, "Individual Tasks", fontsize=10, weight='bold', transform=self.fig_ctrl.transFigure)
        
        ax_btn_tri = plt.axes([0.1, 0.68, 0.35, 0.07])
        self.btn_tri = Button(ax_btn_tri, 'Move Triangle', color='lightgreen', hovercolor='0.9')
        self.btn_tri.on_clicked(lambda x: self.run_task(self.green_triangle))
        
        ax_btn_circ = plt.axes([0.55, 0.68, 0.35, 0.07])
        self.btn_circ = Button(ax_btn_circ, 'Move Circle', color='khaki', hovercolor='0.9')
        self.btn_circ.on_clicked(lambda x: self.run_task(self.yellow_circle))

        # --- Section 3: Blue Square Control ---
        plt.text(0.1, 0.60, "Blue Square Control", fontsize=10, weight='bold', transform=self.fig_ctrl.transFigure)
        
        ax_txt = plt.axes([0.1, 0.52, 0.5, 0.05])
        self.txt_input = TextBox(ax_txt, 'Dest (x y z): ', initial="0 5 0")
        
        ax_btn_sq = plt.axes([0.65, 0.52, 0.25, 0.05])
        self.btn_sq = Button(ax_btn_sq, 'GO', color='lightblue')
        self.btn_sq.on_clicked(self.on_square_submit)

        # --- Section 4: MASTER STACK CONTROL (New) ---
        plt.text(0.1, 0.42, "Master Stack Control (All 3)", fontsize=10, weight='bold', transform=self.fig_ctrl.transFigure)
        
        ax_txt_stack = plt.axes([0.1, 0.34, 0.5, 0.05])
        self.txt_stack = TextBox(ax_txt_stack, 'Target (x y z): ', initial="4 4 0")
        
        ax_btn_stack = plt.axes([0.65, 0.34, 0.25, 0.05])
        self.btn_stack = Button(ax_btn_stack, 'STACK ALL', color='violet')
        self.btn_stack.on_clicked(self.on_stack_all_submit)

        # --- Section 5: Reset ---
        ax_btn_rst = plt.axes([0.1, 0.05, 0.8, 0.1])
        self.btn_rst = Button(ax_btn_rst, 'RESET SIMULATION', color='salmon')
        self.btn_rst.on_clicked(self.reset_sim)

    # --- LOGIC ---
    def get_smart_z(self, x, y, moving_obj_name):
        """Scans (x,y) for existing objects to stack on."""
        max_h = 0.0
        for obj in self.world_objects:
            if obj.name == moving_obj_name: continue
            
            dist = np.sqrt((obj.pos[0]-x)**2 + (obj.pos[1]-y)**2)
            if dist < STACK_TOLERANCE:
                top = obj.pos[2] + obj.height
                if top > max_h: max_h = top
        return max_h

    def on_stack_all_submit(self, event):
        """Moves ALL objects to the specified X,Y using Smart Stacking."""
        try:
            txt = self.txt_stack.text
            v = [float(x) for x in txt.replace(',', ' ').split()]
            if len(v) < 2: return
            
            tx, ty = v[0], v[1]
            tz = v[2] if len(v) > 2 else 0.0
            
            # Check if target is within workspace
            if not self.is_within_workspace(tx, ty, tz):
                print("Outside workspace , cannot reach")
                return
            
            print(f"\n--- EXECUTING MASTER STACK AT ({tx}, {ty}, {tz}) ---")
            
            # Iterate through all objects and move them one by one
            for obj in self.world_objects:
                # Check if object is already at target (x, y)
                dist = np.sqrt((obj.pos[0] - tx)**2 + (obj.pos[1] - ty)**2)
                if dist < STACK_TOLERANCE:
                    print(f"Skipping {obj.name} (Already at target)")
                    continue

                # 1. Calculate the 'Smart Z' for this specific object right now
                # Logic: max(user_z, existing_stack_height)
                stack_top_z = self.get_smart_z(tx, ty, obj.name)
                target_z = max(tz, stack_top_z)
                
                print(f"Moving {obj.name} -> Z={target_z:.2f}")
                self.pick_and_place(obj, tx, ty, target_z)
                
        except ValueError:
            print("Invalid Input for Master Stack")

    def run_task(self, obj):
        if obj.dest is None: return
        
        # Check if destination is within workspace
        if not self.is_within_workspace(obj.dest[0], obj.dest[1], obj.dest[2]):
            print("Outside workspace , cannot reach")
            return
        
        base_z = self.get_smart_z(obj.dest[0], obj.dest[1], obj.name)
        # Requirement: "Z is at level 2/4". If stack is higher, we go higher.
        target_z = max(obj.dest[2], base_z)
        self.pick_and_place(obj, obj.dest[0], obj.dest[1], target_z)

    def on_square_submit(self, event):
        try:
            txt = self.txt_input.text
            v = [float(x) for x in txt.replace(',', ' ').split()]
            if len(v) < 3: return
            
            # Check if destination is within workspace
            if not self.is_within_workspace(v[0], v[1], v[2]):
                print("Outside workspace , cannot reach")
                return
            
            base_z = self.get_smart_z(v[0], v[1], self.blue_square.name)
            target_z = max(v[2], base_z)
            self.pick_and_place(self.blue_square, v[0], v[1], target_z)
        except ValueError:
            print("Invalid Input")

    def is_within_workspace(self, x, y, z):
        """Check if a position is within the robot's reachable workspace."""
        within_x = WORKSPACE_X_MIN <= x <= WORKSPACE_X_MAX
        within_y = WORKSPACE_Y_MIN <= y <= WORKSPACE_Y_MAX
        within_z = WORKSPACE_Z_MIN <= z <= WORKSPACE_Z_MAX
        return within_x and within_y and within_z

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
        t1, t2, t3 = solve(x, y, 0)
        if t1 is not None: return d1, t1, t2, t3
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
        except ValueError: return False

    def pick_and_place(self, obj, dx, dy, dz):
        # EARLY VALIDATION: Check if destination is reachable before lifting object
        try:
            # Test if IK can solve for destination
            self.inverse_kinematics(dx, dy, dz)
        except ValueError:
            print("Outside workspace , cannot reach")
            return
        
        sx, sy, sz = obj.pos
        lift_z = min(sz + SAFE_HOVER_HEIGHT, MAX_Z_HEIGHT)
        drop_hover_z = min(dz + SAFE_HOVER_HEIGHT, MAX_Z_HEIGHT)
        travel_z = max(lift_z, drop_hover_z)
        if not self.move_to(sx, sy, lift_z): return
        if not self.move_to(sx, sy, sz): return
        self.held_object = obj; obj.color = 'red'; time.sleep(0.1)
        if not self.move_to(sx, sy, lift_z): return
        if not self.move_to(dx, dy, travel_z): return
        if not self.move_to(dx, dy, drop_hover_z): return
        if not self.move_to(dx, dy, dz): return
        self.held_object = None; 
        obj.color = 'blue' if 'Square' in obj.name else ('green' if 'Triangle' in obj.name else 'yellow')
        time.sleep(0.1)
        self.move_to(dx, dy, drop_hover_z)

    def draw_scene(self):
        plt.figure(self.fig_viz.number)
        self.ax.cla()
        self.ax.set_xlim(-8, 8); self.ax.set_ylim(-8, 8); self.ax.set_zlim(0, 6)
        self.ax.set_xlabel('X'); self.ax.set_ylabel('Y'); self.ax.set_zlabel('Z')
        self.ax.set_title("Robot Visualization")
        if self.show_workspace:
            X, Y, Z = self.ws_mesh
            self.ax.plot_surface(X, Y, Z, color='yellow', alpha=0.1)

        # --- DRAW ROBOT ---
        p0, p1, p2, p3, p4 = self.forward_kinematics(self.d1, self.theta1, self.theta2, self.theta3)
        
        # Links
        # Link 1 (Prismatic d1): p0 -> p1
        self.ax.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]], 'k-', linewidth=5, label='Link 1')
        # Link 2 (Shoulder): p1 -> p2
        self.ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 'b-', linewidth=5, label='Link 2')
        # Link 3 (Elbow): p2 -> p3
        self.ax.plot([p2[0], p3[0]], [p2[1], p3[1]], [p2[2], p3[2]], 'r-', linewidth=5, label='Link 3')
        # Link 4 (Wrist): p3 -> p4
        self.ax.plot([p3[0], p4[0]], [p3[1], p4[1]], [p3[2], p4[2]], 'g-', linewidth=5, label='Link 4')

        # Joints
        self.ax.scatter([p0[0]], [p0[1]], [p0[2]], c='k', s=100)
        self.ax.scatter([p1[0]], [p1[1]], [p1[2]], c='k', s=100)
        self.ax.scatter([p2[0]], [p2[1]], [p2[2]], c='b', s=80)
        self.ax.scatter([p3[0]], [p3[1]], [p3[2]], c='r', s=80)
        self.ax.scatter([p4[0]], [p4[1]], [p4[2]], c='g', s=80)

        # --- DRAW OBJECTS ---
        for obj in self.world_objects:
            # If object is held, it's at the end-effector (p4)
            if self.held_object == obj:
                pos = p4
            else:
                pos = obj.pos
            
            self.ax.scatter([pos[0]], [pos[1]], [pos[2]], 
                            c=obj.color, marker=obj.marker, s=200, 
                            edgecolors='k', label=obj.name)
            
            # Draw destination ghost if it exists
            if obj.dest is not None:
                self.ax.scatter([obj.dest[0]], [obj.dest[1]], [obj.dest[2]], 
                                c=obj.color, marker=obj.marker, s=50, alpha=0.3)
        
        plt.draw()
        plt.pause(0.001)

# ============================================================================
# MAIN EXECUTION
# ============================================================================
if __name__ == "__main__":
    robot = RobotController()
    plt.show()