import numpy as np
from lyapunovController import LyapunovController
from mpcController import QTO_MPC
from robot import DiffDrive
from estimator import DiffDriveEstimator
from planner import compute_reference_trajectory
import argparse
import yaml
from visualize import visualize, plot

##### KHALED WAHBA AUTHOR
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", type=str, default="problems/empty.yaml", help="path to problem and robot description yaml file")
    parser.add_argument("--output", type=str, default="simulation_visualization", help="path to output visualization pdf and html file")
    args = parser.parse_args()

    # Initialize robot model
    problem_path = args.problem
    with open(problem_path, 'r') as file:
        problem = yaml.safe_load(file)

    # Simulation parameters
    sim_time = problem["sim_time"]  # simulation duration (s)
    dt = float(problem["time_step"])
    sim_N = int(sim_time / dt)
    sim_time_grid = np.linspace(0, sim_N * dt, sim_N + 1)
    start = problem["start"]
    goal = problem["goal"]
    min_distance_to_goal = problem["controllers"]["mpc"]["min_distance_to_goal"]

    # Initialize robot and estimator
    robot_cfg = problem["robotcfg"]
    robot = DiffDrive(robot_cfg=robot_cfg, init_state=start, dt=dt)
    # Planning time horizon
    planner_cfg = problem["planner"]
    planner_time = planner_cfg["time"]  # total time for reference trajectory (s)
    planner_N = int(planner_time / dt)
    planner_time_grid = np.linspace(0, planner_N * dt, planner_N + 1)
    
    # Generate reference trajectory over full planner horizon
    waypoints = planner_cfg["waypoints"]
    reference_states, polynomial_traj = compute_reference_trajectory(start, goal, waypoints, planner_time_grid)

    # Initialize estimator
    est_cfg = problem["estimator"]  # may be empty
    # Estimator uses robot+estimator params
    estimator = DiffDriveEstimator(estimator_cfg=est_cfg,
                                dt=dt)    # simulation time horizon
    # Initialize controller(s)
    controllers = problem["controllers"]
    controller_objects = []
    for controller_key, controller_item in controllers.items():
        if "lyapunov" in controller_key:
            ctrl = LyapunovController(robot_param=est_cfg,
                            gains=controller_item["gains"],
                            cmd_limits=[robot_cfg["min_vel_rightwheel"], robot_cfg["max_vel_rightwheel"], robot_cfg["min_vel_leftwheel"], robot_cfg["max_vel_leftwheel"]],
                            dt=dt)
            controller_objects.append(ctrl)
        elif "mpc" in controller_key:
            qto_mpc = QTO_MPC(robot_param=problem) # TODO: fix arguments here
            # mpc_params = controller_item
            controller_objects.append(qto_mpc)
            # # TODO: implement MPC controller initialization
            pass
        elif "realtime_dbA" in controller_key:
            #TODO: implement realtime dBA controller initialization
            pass
        else:
            raise ValueError("No valid controller found in problem description.")

    # Simulate only for the simulation time horizon for all controllers
    for controller in controller_objects:    
        for k in range(len(sim_time_grid)):
            # Update estimator with true wheel speeds
            ur_true, ul_true = robot.get_wheel_speeds()
            pose_true = robot.get_pose()
            estimator.update(ur_true, ul_true, pose_true)
            pose_est = estimator.get_est_pose()
            ur_hat, ul_hat = estimator.get_est_wheel_speeds()
            wheel_est = (ur_hat, ul_hat)
            # Determine which reference state to use
            if k < len(reference_states):
                # Use current reference state if available
                ref_state = reference_states[k]
            else:
                # Use last available reference state if simulation continues beyond planner time
                ref_state = reference_states[-1]

            # Compute control commands [ur_cmd, ul_cmd] using reference at current time step
            if controller.name == "lyapunov":
                u = controller.compute(ref_state, pose_est, wheel_est)
            elif controller.name == "mpc":
                if qto_mpc.goal_reached(pose_true, goal, min_distance_to_goal):
                    print(40 * "-")
                    print(f"Goal reached at time {k*dt:.2f} seconds.")
                    print(40 * "-")
                    break
                u = qto_mpc.solve(pose_true, goal)
            elif controller.name == "realtime_dbA":
                pass
            else:
                raise ValueError("Unknown controller type during simulation.")
            # Step the robot simulation
            robot.step(u)
            
        # Plot results for this controller
        plot(robot, estimator, reference_states, sim_time_grid, out_prefix=args.output+"_"+controller.name)
        visualize(problem_path, robot, reference_states, out_prefix=args.output+"_"+controller.name)
