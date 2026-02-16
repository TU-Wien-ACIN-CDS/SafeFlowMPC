import argparse
import time
from glob import glob
from pathlib import Path

import numpy as np

from safe_flow_mpc.RobotModel import RobotModel
from safe_flow_mpc.SafeFlowMPC import PlannerConfig, SafeFlowMPC
from safe_flow_mpc.SafeFlowMPC.ObstacleManager import ObstacleManager


def load_trajectory_data(config: PlannerConfig, file_idx: int = None):
    """Load trajectory data from file."""

    print(f"Loading file {file_idx} ...")
    data_file = Path(config.data_path) / f"{config.data_name}{file_idx}.npz"
    data = np.load(data_file, allow_pickle=True)

    q0 = data["q"][0, :]
    qf = data["q"][-1, :]
    q_ref = data["q"]
    q_prev = data["q_prev0"].tolist()
    n_timesteps = data["q"].shape[0]

    return q0, qf, q_ref, q_prev, n_timesteps


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Global Planner Inference")
    parser.add_argument(
        "--replan", action="store_true", help="Enable replanning", default=False
    )
    args = parser.parse_args()

    # Configuration
    config = PlannerConfig(
        use_safety_filter=True,
        use_safe_model=True,
        use_sets=True,
        use_term=True,
        n_horizon=16,
        flow_steps=7,
        compile_fm=False,
        build=False,
        fm_dim=32,
        fm_dim_mults=(1, 2, 4, 8),
        real_time=False,
        data_path="./data/",
        experiment=False,
        limit_time=False,
        model_path="checkpoints/",
    )

    # Load trajectory data
    rng = np.random.default_rng()
    robot_model = RobotModel()
    nr_files = len(glob("data/traj_example_*.npz"))
    file_idx = rng.integers(0, nr_files, (1,))[0]
    q_start, q_goal, q_ref, q_prev0, n_timesteps = load_trajectory_data(
        config, file_idx
    )
    p_goal, _, _ = robot_model.forward_kinematics(q_goal, 0 * q_goal)

    # Create planner
    obstacle_manager = ObstacleManager()
    obstacle_manager.add_default_obstacles()
    # obstacle_manager.add_oriented_box(p_goal, np.array([0.05, 0.03, 0.2]))
    planner = SafeFlowMPC(
        config,
        obstacle_manager,
        workspace_max=[1.3, 1.3, 1.5],
        workspace_min=[-0.25, -1.3, 0.1],
    )
    planner.state.q = q_start
    planner.state.q_prev = q_prev0

    # Plan trajectory
    # Initialize
    planner.set_start_and_goal(q_start, p_goal)
    if not np.any(planner.state.q):
        planner.set_state()
    planner.x_current = planner.sample_initial_trajectory()

    # Move initial position
    if planner.config.experiment:
        planner.robot_controller.move_initial_position(planner.q_start)
        planner.robot_controller.set_init_time()

    # Planning loop
    replanning = args.replan
    replan_times = rng.integers(0, 50, 3)
    for timestep in range(n_timesteps + 50):
        if timestep in replan_times and replanning:
            print("Replanning")
            # Adapt the goal
            file_idx = rng.integers(0, nr_files, (1,))[0]
            q_start, q_goal, q_ref, q_prev0, n_timesteps = load_trajectory_data(
                config, file_idx
            )
            p_goal, _, _ = robot_model.forward_kinematics(q_goal, 0 * q_goal)
            planner.p_goal = p_goal
        planner.step()
        if planner.metrics["success"]:
            break

    time.sleep(1)
    print("Closing viewer")
    planner.viewer.close()


if __name__ == "__main__":
    main()
