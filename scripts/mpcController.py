import os
import glob
import shutil
import yaml
import numpy as np
import casadi as ca
import matplotlib.pyplot as plt

class QTO_MPC():
    """Quasi Time-Optimal Model Predictive Control (QTO-MPC) class."""
    
    def __init__(self, robot_param):
        environment = robot_param["environment"]
        cmd_limits = [robot_param["robotcfg"]["min_vel_leftwheel"], robot_param["robotcfg"]["max_vel_leftwheel"], robot_param["robotcfg"]["min_vel_rightwheel"], robot_param["robotcfg"]["max_vel_rightwheel"]]
        dt = robot_param["time_step"]
        # MPC parameters
        self.N = 20     # Prediction horizon
        self.dt = dt   # Sampling time [s]
        self.nx = 3     # Number of states
        self.nu = 2     # Number of controls
        self.name = "mpc"
        # Robot parameters
        self.wheel_radius = robot_param["robotcfg"]["wheel_radius"] 
        self.distance_between_wheels = robot_param["robotcfg"]["base_diameter"] 

        # State limits
        x_min = [environment["min"][0], environment["min"][1], -ca.inf]
        x_max = [environment["max"][0], environment["max"][1], ca.inf]

        # Control limits
        self.cmd_limits = cmd_limits
        umin_right, umax_right, umin_left, umax_left = self.cmd_limits

        u_min = [umin_right, umin_left]
        u_max = [umax_right, umax_left]
        
        # Cost function weights
        Q = ca.diag([10.0, 10.0, 0.1])
        QN = ca.diag([10.0, 10.0, 0.1])
        R = ca.diag([0.0, 0.0])

        # Initial and reference states
        x_init = ca.MX.sym("x_init", self.nx)
        x_ref = ca.MX.sym("x_ref", self.nx)

        # OCP formulation
        X = []
        self.lbX = []
        self.ubX = []
        G = []
        self.lbG = []
        self.ubG = []
        P = []
        J = 0

        for k in range(self.N):
            x = ca.MX.sym(f"x_{k}", self.nx)
            X.append(x)
            self.lbX.extend(x_min)
            self.ubX.extend(x_max)

            G.append(
                x - x_init
                if k == 0
                else x - self.robot_dynamics(x_prev, u_prev, self.dt)  # noqa: F821
            )
            self.lbG.extend(self.nx * [0])
            self.ubG.extend(self.nx * [0])

            u = ca.MX.sym(f"u_{k}", self.nu)
            X.append(u)
            self.lbX.extend(u_min)
            self.ubX.extend(u_max)

            J += (x - x_ref).T @ Q @ (x - x_ref) + u.T @ R @ u

            x_prev = x
            u_prev = u

        x = ca.MX.sym(f"x_{self.N}", self.nx)

        X.append(x)
        self.lbX.extend(x_min)
        self.ubX.extend(x_max)

        G.append(x - self.robot_dynamics(x_prev, u_prev, self.dt))
        self.lbG.extend(self.nx * [0])
        self.ubG.extend(self.nx * [0])

        J += (x - x_ref).T @ QN @ (x - x_ref)

        # Declare NLP parameters
        P.extend([x_init, x_ref])

        # Define NLP problem
        nlp = {"f": J, "x": ca.vertcat(*X), "g": ca.vertcat(*G), "p": ca.vertcat(*P)}
        casadi_tmp_dir = "casadi_tmp/"
        os.makedirs(casadi_tmp_dir, exist_ok=True)
        opts = {
            "ipopt": {
                "print_level": 4,
                "max_wall_time": self.dt,
                "warm_start_init_point": "yes",
                "linear_solver": "mumps",
            },
            "jit": True,
            "jit_options": {
                "flags": ["-O3", "-ffast-math"],
                "verbose": True,
                "compiler": "gcc",
                "directory": casadi_tmp_dir,
            }
        }

        print("Compiling NLP solver...")
        self.solver = ca.nlpsol("solver", "ipopt", nlp, opts)
        print("NLP solver compiled successfully.")

        [os.remove(filename) for filename in glob.glob("*.c")]
        shutil.rmtree(casadi_tmp_dir, ignore_errors=True)

        self.initiaize_variables()

    def goal_reached(self, x, goal, min_dist):
        dist = np.linalg.norm(np.array(x[:2]) - np.array(goal[:2])) + np.abs(x[2] - goal[2])
        return dist < min_dist

    def robot_dynamics(self, x, u, dt):
        """Robot dynamics for a differential drive robot."""
        v_left, v_right = u[0], u[1]
        v = self.wheel_radius * (v_left + v_right) / 2.0
        omega = self.wheel_radius * (v_right - v_left) / self.distance_between_wheels

        x_next = ca.vertcat(
            x[0] + v * ca.cos(x[2]) * dt,
            x[1] + v * ca.sin(x[2]) * dt,
            x[2] + omega * dt
        )
        return x_next
    
    def initiaize_variables(self):
        """Initialize optimization variables."""
        self.x_opt = np.zeros(len(self.lbX))
        self.lam_x_opt = np.zeros(len(self.lbX))
        self.lam_g_opt = np.zeros(len(self.lbG))

    def solve(self, x_init, x_ref):
        """Solve the MPC optimization problem."""
        P = []
        P.append(x_init)
        P.append(x_ref)
        P = ca.vertcat(*P)

        opt_sol = self.solver(
            x0=self.x_opt,
            lam_x0=self.lam_x_opt,
            lam_g0=self.lam_g_opt,
            lbx=self.lbX,
            ubx=self.ubX,
            lbg=self.lbG,
            ubg=self.ubG,
            p=P,
        )

        self.x_opt = opt_sol["x"].full().flatten()
        self.lam_x_opt = opt_sol["lam_x"]
        self.lam_g_opt = opt_sol["lam_g"]

        u_opt = self.x_opt[self.nx + 0:self.nx + 2]
        return u_opt
        # Define helper functions and the MPC planner


# wheel_radius = config["wheel_radius"] / 1000.0  # Convert mm to m
# wheel_radius = robot_param["robotcfg"]["wheel_radius"]
# wheel_radius = self.wheel_radius
# distance_between_wheels = robot_param["robotcfg"]["base_diameter"]

def simulator(x, u, dt=0.01):
    v_left, v_right = u[0], u[1]
    v = wheel_radius * (v_left + v_right) / 2.0
    omega = wheel_radius * (v_right - v_left) / distance_between_wheels

    x_next = np.array(
        [
            x[0] + v * np.cos(x[2]) * dt,
            x[1] + v * np.sin(x[2]) * dt,
            x[2] + omega * dt
        ]
    )
    return x_next

def goal_reached(x, goal, min_dist):
    dist = np.linalg.norm(np.array(x[:2]) - np.array(goal[:2])) + np.abs(x[2] - goal[2])
    return dist < min_dist

# qto_mpc = QTO_MPC(config)
# # Run simulation

# start = robot_param["start"]
# goal = robot_param["goal"]
# min_distance_to_goal = corobot_paramnfig["controllers"]["mpc"]["min_distance_to_goal"]
# time_step = robot_param["time_step"]
# N_steps = int(robot_param["sim_time"]/robot_param["time_step"])

# x_traj = np.zeros((N_steps, 3))
# u_traj = np.zeros((N_steps, 2))

# x_traj[0] = start
# qto_mpc.initiaize_variables()

# for i in range(N_steps-1):
#     if goal_reached(x_traj[i], goal, min_distance_to_goal):
#         print(40 * "-")
#         print(f"Goal reached at time {i*time_step:.2f} seconds.")
#         print(40 * "-")
#         x_traj[i + 1:] = x_traj[i]
#         break

#     u_opt = qto_mpc.solve(x_traj[i], goal)
#     u_traj[i] = u_opt
#     x_traj[i + 1] = simulator(x_traj[i], u_opt, dt=time_step)

#     # Visualize results

# def plot_robot(ax, x, y, theta, color="r", size=0.3, alpha=0.7):
#     triangle = np.array([
#         [size, 0],
#         [-size/2, size/2],
#         [-size/2, -size/2]
#     ])
#     R = np.array([
#         [np.cos(theta), -np.sin(theta)],
#         [np.sin(theta),  np.cos(theta)]
#     ])
#     tri_world = (R @ triangle.T).T + np.array([x, y])
#     ax.fill(tri_world[:, 0], tri_world[:, 1], color=color, alpha=alpha)

# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# # Plot trajectory
# ax1.plot(x_traj[:, 0], x_traj[:, 1], label="Trajectory")
# ax1.plot(start[0], start[1], "ko", label="Start")
# ax1.plot(goal[0], goal[1], "go", label="Goal")
# for t in range(int(N_steps * time_step)):
#     idx = min(int(t / time_step), x_traj.shape[0]-1)
#     plot_robot(ax1, x_traj[idx, 0], x_traj[idx, 1], x_traj[idx, 2], color="r")
# ax1.set_xlabel("x [m]")
# ax1.set_ylabel("y [m]")
# ax1.set_title("Trajectory of the Robot")
# ax1.set_aspect("equal")
# ax1.set_xlim([-7, 7])
# ax1.set_ylim([-7, 7])
# ax1.legend()
# ax1.grid()

# # Plot control signals
# time_array = np.arange(N_steps) * time_step
# ax2.plot(time_array, u_traj[:, 0], label="Left Wheel Velocity")
# ax2.plot(time_array, u_traj[:, 1], label="Right Wheel Velocity")
# ax2.set_xlabel("t [s]")
# ax2.set_ylabel("Wheel Velocity [rad/s]")
# ax2.legend()
# ax2.grid()
# ax2.set_title("Control Signals")

# plt.tight_layout()
# plt.show()
