import numpy as np
import matplotlib.pyplot as plt

def dh_transform(a, alpha, d, theta):
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [ct, -st*ca,  st*sa, a*ct],
        [st,  ct*ca, -ct*sa, a*st],
        [0,      sa,     ca,    d],
        [0,       0,      0,    1]
    ])

# Robot parameters
L1 = 3.0
L2 = 5.0
P0 = np.array([0, 0, 0, 1])

# --------------------------------------------------
# SET UP ONE WINDOW
# --------------------------------------------------
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Create line objects for both links
line1, = ax.plot([], [], [], 'b-', linewidth=6)   # Link 1
line2, = ax.plot([], [], [], 'g-', linewidth=6)   # Link 2

# Base joint marker
ax.scatter(0, 0, 0, s=150, color='red')

ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10)
ax.set_zlim(0, 10)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("RR Robot – θ1 & θ2 Animation (Same Frame)")

plt.ion()
plt.show()

# --------------------------------------------------
# ANIMATION LOOP — BOTH θ1 AND θ2 INCREMENT TOGETHER
# --------------------------------------------------
for step in range(360):  # 0° to 350°
    
    theta1 = np.radians(step * 0)
    theta2 = np.radians(step + 10)

    # First transformation (base to joint 2)
    T01 = dh_transform(a=0, alpha=np.pi/2, d=L1, theta=theta1)
    P1 = (T01 @ P0)[:3]

    # Second transformation (joint2 to end-effector)
    T12 = dh_transform(a=L2, alpha=0, d=0, theta=theta2)
    P2 = (T01 @ T12 @ P0)[:3]

    # UPDATE LINK 1
    line1.set_data([0, P1[0]], [0, P1[1]])
    line1.set_3d_properties([0, P1[2]])

    # UPDATE LINK 2
    line2.set_data([P1[0], P2[0]], [P1[1], P2[1]])
    line2.set_3d_properties([P1[2], P2[2]])

    # Update title dynamically
    ax.set_title(f"RR Robot – θ1 = θ2 = {step*10}°")

    plt.pause(0.15)

plt.ioff()
