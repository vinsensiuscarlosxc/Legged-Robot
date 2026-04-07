import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def run_inverse():

    # =========================
    # PARAMETER ROBOT (3 DOF)
    # =========================
    L1, L2, L3 = 8, 6, 4

    # target point
    targets = [(8,5),(10,2),(5,10),(-6,8),(-8,-2),(4,-6)]

    x_traj, y_traj = [], []

    # =========================
    # TRAJECTORY (TETAP)
    # =========================
    for i in range(len(targets)):
        x0, y0 = targets[i]
        x1, y1 = targets[(i+1) % len(targets)]

        t = np.linspace(0, 1, 80)

        x_traj.extend(x0 + (x1 - x0) * t)
        y_traj.extend(y0 + (y1 - y0) * t)

    # =========================
    # SETUP VISUAL (IMPROVED)
    # =========================
    plt.style.use('dark_background')

    fig, ax = plt.subplots(figsize=(7,7))

    ax.set_xlim(-20,20)
    ax.set_ylim(-20,20)
    ax.set_aspect('equal')

    ax.grid(True, linestyle='--', alpha=0.3)

    # workspace
    circle = plt.Circle((0,0), L1+L2+L3,
                        fill=False, linestyle="--",
                        linewidth=1.5, alpha=0.4)
    ax.add_patch(circle)

    # robot links (beda warna)
    line1, = ax.plot([], [], 'o-', lw=5)
    line2, = ax.plot([], [], 'o-', lw=5)
    line3, = ax.plot([], [], 'o-', lw=5)

    traj, = ax.plot([], [], lw=2)
    target_point, = ax.plot([], [], 'o', markersize=8)

    text = ax.text(0, -18, '',
                   ha='center',
                   fontsize=10,
                   bbox=dict(facecolor='black',
                             alpha=0.6,
                             edgecolor='white'))

    x_hist, y_hist = [], []

    # =========================
    # UPDATE
    # =========================
    def update(i):

        x = x_traj[i]
        y = y_traj[i]

        # =========================
        # INVERSE KINEMATICS
        # =========================
        phi = 0.3 * np.sin(i * 0.03)

        xw = x - L3 * np.cos(phi)
        yw = y - L3 * np.sin(phi)

        D = (xw**2 + yw**2 - L1**2 - L2**2) / (2 * L1 * L2)
        D = np.clip(D, -1, 1)

        theta2 = np.arccos(D)

        theta1 = np.arctan2(yw, xw) - np.arctan2(
            L2*np.sin(theta2),
            L1 + L2*np.cos(theta2)
        )

        theta3 = phi - theta1 - theta2

        # =========================
        # FORWARD KINEMATICS
        # =========================
        x1 = L1*np.cos(theta1)
        y1 = L1*np.sin(theta1)

        x2 = x1 + L2*np.cos(theta1+theta2)
        y2 = y1 + L2*np.sin(theta1+theta2)

        x3 = x2 + L3*np.cos(theta1+theta2+theta3)
        y3 = y2 + L3*np.sin(theta1+theta2+theta3)

        # =========================
        # DRAW ROBOT
        # =========================
        line1.set_data([0,x1],[0,y1])
        line1.set_color('cyan')

        line2.set_data([x1,x2],[y1,y2])
        line2.set_color('yellow')

        line3.set_data([x2,x3],[y2,y3])
        line3.set_color('magenta')

        # trajectory
        x_hist.append(x3)
        y_hist.append(y3)
        traj.set_data(x_hist, y_hist)
        traj.set_color('orange')

        # target
        target_point.set_data([x], [y])
        target_point.set_color('red')

        # text info
        text.set_text(
            f"Inverse Kinematics\n"
            f"Target: ({x:.2f}, {y:.2f})\n"
            f"End Effector: ({x3:.2f}, {y3:.2f})"
        )

        return line1, line2, line3, traj, target_point, text

    # =========================
    # ANIMATION
    # =========================
    anim = FuncAnimation(
        fig,
        update,
        frames=len(x_traj),
        interval=40,
        repeat=True
    )

    plt.title("3 DOF Robot - Inverse Kinematics (Enhanced)",
              fontsize=13, weight='bold')

    plt.xlabel("X Position")
    plt.ylabel("Y Position")

    plt.show()


# RUN
run_inverse()