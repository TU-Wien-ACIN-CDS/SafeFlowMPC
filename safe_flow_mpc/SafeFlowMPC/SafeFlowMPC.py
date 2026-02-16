import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import casadi as ca
import matplotlib.pyplot as plt
import mujoco
import mujoco.viewer
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

from safe_flow_mpc.SafetyFilter import (
    SafetyFilter,
    SafetyFilterAcados,
)

from ..RobotModel import RobotModel
from ..utils import projection_opt_problem
from .FlowMatchingField import FlowMatchingField
from .ObstacleManager import ObstacleManager
from .PlannerConfig import PlannerConfig
from .SimulationState import SimulationState


def guidance_problem():
    jac = ca.SX.sym("jac", 6, 7)
    dpose = ca.SX.sym("dpose", 6)
    dq = ca.SX.sym("dq", 7)

    params = ca.vertcat(jac.reshape((-1, 1)), dpose)

    g = []
    lbg = []
    ubg = []
    u = []
    lbu = []
    ubu = []

    u += [dq]
    lbu += [-np.pi / 4] * 7
    ubu += [np.pi / 4] * 7

    J = ca.sumsqr(dpose - jac @ dq)

    prob = {"f": J, "x": ca.vertcat(*u), "g": ca.vertcat(*g), "p": params}

    solver = ca.qpsol(
        "solver",
        "osqp",
        prob,
        {
            "print_time": False,
            "osqp": {
                "verbose": False,
                "eps_abs": 1e-6,
                "eps_rel": 1e-6,
            },
        },
    )
    return solver, lbu, ubu, lbg, ubg


class SafeFlowMPC:
    """Main trajectory planning class integrating all components."""

    def __init__(
        self,
        config: PlannerConfig,
        obstacle_manager: ObstacleManager,
        workspace_max=[2, 2, 2],
        workspace_min=[-2, -2, -2],
    ):
        self.config = config
        self.device = self._setup_device()
        self.rng = np.random.default_rng()
        self.last_safe_trajectory = None
        self.time_limit = 0.8 * self.config.dt_sim
        self.workspace_max = workspace_max
        self.workspace_min = workspace_min
        self.state = SimulationState(np.zeros(7))
        self.t_gripper_open = np.inf

        # Initialize components
        self.robot_model = RobotModel()
        self.obstacle_manager = obstacle_manager
        self.flow_model = FlowMatchingField(config, self.device)

        # Initialize safety filters
        self.initialize_safety_filters()

        # Initialize collision checking solver
        self.collision_solver, _, _, self.lbg, self.ubg = projection_opt_problem(
            max_set_size=30
        )

        # Initialize mujoco
        # self.model = mujoco.MjModel.from_xml_path("mujoco_model/mujoco_env.xml")
        self.model = mujoco.MjModel.from_binary_path(
            "mujoco_model/mujoco_env_inference.mjb"
        )
        self.data = mujoco.MjData(self.model)
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

        # Logging and metrics
        self.metrics = {
            "positions": [],
            "timesteps": 0,
            "goal_distance": 0.0,
            "velocity_norm": 0.0,
            "success": False,
            "times": [],
            "flow_times": [],
            "opt_times": [],
            "safety_interventions": [],
            "collision_distances": [],
            "x_plot": [],  # Joint trajectories
            "dx_plot": [],  # Joint velocities
            "ddx_plot": [],  # Joint accelerations
            "dddx_plot": [],  # Joint jerks
            "x_steps": [],  # Flow matching steps
        }

    def _setup_device(self) -> str:
        """Setup computation device (GPU/CPU)."""
        if torch.cuda.is_available():
            print("Using GPU")
            return "cuda:0"
        else:
            print("Using CPU")
            return "cpu"

    def set_handover_data(self, p_receiver):
        print("Setting handover data...")
        self.p_receiver = p_receiver

    def initialize_safety_filters(self) -> None:
        """Initialize safety filtering components."""
        self.safety_filter = SafetyFilterAcados(
            N=self.config.n_horizon,
            smooth=self.config.smooth,
            use_term=self.config.use_term,
            use_sets=self.config.use_sets,
            handover=self.config.handover,
            obstacle_manager=self.obstacle_manager,
            build=self.config.build,
            workspace_max=self.workspace_max,
            workspace_min=self.workspace_min,
        )

        self.safety_filter_init = SafetyFilter(
            N=self.config.n_horizon,
            smooth=self.config.smooth,
            use_term=self.config.use_term,
            use_sets=False,
            obstacle_manager=self.obstacle_manager,
        )

    def load_trajectory_data(
        self, file_idx: int = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Load trajectory data from file."""

        print(f"Loading file {file_idx} ...")
        data_file = (
            Path(self.config.data_path) / f"{self.config.data_name}{file_idx}.npz"
        )
        data = np.load(data_file, allow_pickle=True)

        q0 = data["q"][0, :]
        qf = data["q"][-1, :]
        n_timesteps = data["q"].shape[0]

        return q0, qf, n_timesteps

    def create_condition_vector(self) -> torch.Tensor:
        """Create condition vector for neural network."""
        q_prev_tensors = [torch.Tensor(x).to(self.device) for x in self.state.q_prev]

        # Current end-effector pose
        h = self.robot_model.hom_transform_endeffector(self.state.q)
        p0 = h[:3, 3]
        r0 = h[:3, :3].flatten()
        if self.config.handover:
            if self.config.experiment:
                p_receiver_prev_tensors = [
                    torch.Tensor(x).to(self.device) for x in self.p_receiver
                ]
            else:
                idx_prev = [self.state.timestep - x for x in reversed(range(10))]
                idx_prev = np.maximum(idx_prev, 0)
                idx_prev = np.minimum(idx_prev, self.p_receiver.shape[0] - 1)
                p_receiver_prev_tensors = [
                    torch.Tensor(x).to(self.device)
                    for x in self.p_receiver[idx_prev, :]
                ]
            condition = torch.cat(
                q_prev_tensors
                + [torch.Tensor(p0).to(self.device)]
                + [torch.Tensor(r0).to(self.device)]
                + p_receiver_prev_tensors
            )
        else:
            # Target end-effector pose
            r_final = R.from_rotvec(self.p_goal[3:]).as_matrix().flatten()

            # Collision sphere positions
            p_cols = [
                torch.Tensor(self.robot_model.fk_pos_col(self.state.q, i)).to(
                    self.device
                )
                for i in range(7)
            ]

            condition = torch.cat(
                q_prev_tensors
                + [torch.Tensor(p0).to(self.device)]
                + [torch.Tensor(r0).to(self.device)]
                + [torch.Tensor(self.p_goal[:3]).to(self.device)]
                + [torch.Tensor(r_final).to(self.device)]
                + p_cols
            )

        return condition

    def sample_initial_trajectory(self) -> np.ndarray:
        """Sample initial trajectory for planning."""
        if self.config.use_safe_dist:
            jerk_0 = np.zeros((self.config.n_horizon, 7))  # Simplified sampling
            x_0 = self._jerk_to_pos(
                jerk_0, self.state.q, self.state.dq, self.state.ddq, self.state.dddq
            )

            # Apply joint limits
            x_0 = np.clip(
                x_0, self.robot_model.q_lim_lower, self.robot_model.q_lim_upper
            )

            # Apply safety filter
            if self.safety_filter_init:
                x_0 = self.safety_filter_init.step(self.state.q, x_0, update=False).T

            x_0 = torch.Tensor(x_0.T).to(self.device)
            x_0_rest = torch.randn((self.config.n_out - 7, self.config.n_horizon)).to(
                self.device
            )
            x_0 = torch.cat((x_0, x_0_rest), dim=0).T.reshape((1, -1))

            return x_0
        else:
            return (
                torch.randn((self.config.n_horizon, self.config.n_out))
                .reshape((1, -1))
                .to(self.device)
            )

    def update_current_solution(self):
        if self.config.use_safety_filter:
            x_new = self.safety_filter.q_last[:, 1:]
            x_new = torch.Tensor(x_new.T).to(self.device)
            # x_old = self.x_current.reshape((self.config.n_horizon, self.config.n_out))
            x_new_rest = (
                torch.randn((self.config.n_out - 7, self.config.n_horizon))
                .to(self.device)
                .T
            )
            x_new = torch.cat((x_new, x_new_rest), dim=1).reshape((1, -1))
        else:
            x_new = self.sample_initial_trajectory()
        return x_new

    def _jerk_to_pos(
        self,
        u: np.ndarray,
        q0: np.ndarray,
        dq0: np.ndarray,
        ddq0: np.ndarray,
        dddq0: np.ndarray,
        dt: float = 0.1,
    ) -> np.ndarray:
        """Convert jerk commands to position trajectory."""
        u = u.reshape((u.shape[0], 7))
        q = np.zeros_like(u)
        dq = np.zeros_like(u)
        ddq = np.zeros_like(u)

        q[0, :] = q0
        dq[0, :] = dq0
        ddq[0, :] = ddq0
        u[0, :] = dddq0

        for k in range(q.shape[0] - 1):
            q[k + 1, :] = (
                ddq[k, :] * dt**2 / 2.0
                + dq[k, :] * dt
                + q[k, :]
                + u[k, :] * dt**3 / 8.0
                + u[k + 1, :] * dt**3 / 24.0
            )
            dq[k + 1, :] = (
                ddq[k, :] * dt
                + dq[k, :]
                + u[k, :] * dt**2 / 3.0
                + u[k + 1, :] * dt**2 / 6.0
            )
            ddq[k + 1, :] = ddq[k, :] + u[k, :] * dt / 2.0 + u[k + 1, :] * dt / 2.0

        return q

    def set_start_and_goal(self, q_start, p_goal):
        self.q_start = q_start
        self.p_goal = p_goal

    def set_state(self):
        self.state = SimulationState(q=self.q_start.copy())

    def plan_trajectory(
        self,
        q_start: np.ndarray,
        p_goal: np.ndarray,
        n_timesteps: int = 50,
    ) -> Dict[str, Any]:
        """Main trajectory planning function."""

        # Initialize
        self.set_start_and_goal(q_start, p_goal)
        if not np.any(self.state.q):
            self.set_state()
        self.x_current = self.sample_initial_trajectory()

        # Move initial position
        if self.config.experiment:
            self.robot_controller.move_initial_position(self.q_start)
            self.robot_controller.set_init_time()

        # solver, lbu, ubu, lbg, ubg = guidance_problem()

        # Planning loop
        for timestep in range(n_timesteps):
            self.step()
            if self.metrics["success"]:
                break

        # Keep sending the rest of the current horizon
        if self.config.experiment:
            for i in range(1, 10):
                jerk_traj = self.safety_filter.dddq_last[:, i:]
                self.robot_controller.send_joint_jerk_trajectory(jerk_traj)
        self.metrics["timesteps"] = self.state.timestep
        return self.metrics

    def step(self):
        if self.state.timestep > 0:
            self.x_current = self.update_current_solution()

        # Create condition for neural network
        condition = self.create_condition_vector()

        # Flow matching steps
        safety_intervention = 0.0
        x_steps_current = [self.x_current.detach().cpu().numpy()]  # Store initial state

        flow_time = 0.0
        t_start = time.perf_counter()
        sf_converged = False
        dt = 1.0 / self.config.flow_steps
        for flow_step in range(self.config.flow_steps):
            t = (flow_step / self.config.flow_steps) * torch.ones(1).to(self.device)

            if (
                self.config.use_safety_filter
                and time.perf_counter() - t_start > self.time_limit
                and self.config.limit_time
            ):
                print(
                    f"Time limit exceeded in t={self.state.timestep / 10:.3f} step {flow_step}, {time.perf_counter() - t_start:.3f} > {self.time_limit:.3f}, falling back to last safe trajectory"
                )
                if self.last_safe_trajectory is not None:
                    self.x_current = self.last_safe_trajectory
                    break

            # Compute neural velocity field
            t_flow_start = time.perf_counter()
            dx_flow = self.flow_model.compute_velocity(self.x_current, t, condition, dt)
            flow_time += time.perf_counter() - t_flow_start

            # Apply safety filter if enabled
            if self.config.use_safety_filter and self.safety_filter:
                q_des = self._compute_guidance(
                    self.x_current, dx_flow, self.q_start, self.p_goal
                )
                try:
                    q_safe = self.safety_filter.step(self.state.q, q_des)
                    # safety_intervention += np.linalg.norm(q_des - q_safe.T)
                    sf_converged = True
                    x_new = torch.Tensor(q_safe.T).to(self.device)
                    x_old = (self.x_current + dx_flow).reshape(
                        (self.config.n_horizon, self.config.n_out)
                    )
                    self.x_current = torch.cat((x_new, x_old[:, 7:]), dim=1).reshape(
                        (1, -1)
                    )
                except RuntimeError as e:
                    print(
                        f"[Warning] No convergence of SQP in t={self.state.timestep / 10:.3f}, flow_step={flow_step}/{self.config.flow_steps}"
                    )
                    print(e)
                    x_new = self.safety_filter.q_last[:, :-1]
                    x_new = torch.Tensor(x_new.T).to(self.device)
                    x_old = (self.x_current + dx_flow).reshape(
                        (self.config.n_horizon, self.config.n_out)
                    )
                    self.x_current = torch.cat((x_new, x_old[:, 7:]), dim=1).reshape(
                        (1, -1)
                    )
            else:
                self.x_current += dx_flow

            self.last_safe_trajectory = self.x_current.detach()

            # Store flow step results
            if not self.config.real_time:
                x_steps_current.append(self.x_current.detach().cpu().numpy())
            # time.sleep(1)
            # self._update_visualization()

            # print(f"Flow time: {flow_time}")
            # print(f"Sets time: {self.safety_filter.t_sets}")
            # print(f"Opt time: {self.safety_filter.t_array}")
            # print(f"Opt loop: {self.safety_filter.t_loop}")
            # print(time.perf_counter() - t_start)
        if not sf_converged and self.config.use_safety_filter:
            self.safety_filter.update_from_last_solution(self.config.n_actions)
            xc = self.x_current.reshape((self.config.n_horizon, self.config.n_out))
            xc[:, :7] = torch.Tensor(self.safety_filter.q_last[:, :-1].T).to(
                self.device
            )
            self.x_current = xc.reshape((1, -1))

        # Send trajectory to the robot
        if self.config.experiment:
            jerk_traj = self.safety_filter.dddq_last
            q_traj = self.safety_filter.q_last
            self.robot_controller.send_joint_jerk_trajectory(jerk_traj, q_traj)

        # Calculate collision distances
        if not self.config.real_time:
            self.calculate_collision_distances(self.state)

        # Update state and metrics
        self._update_state_and_metrics(
            time.perf_counter() - t_start,
            flow_time,
            safety_intervention,
            x_steps_current,
        )

        # Visualization
        self._update_visualization()

        success = False
        dist_goal = np.linalg.norm(
            self.p_goal[:3] - self.robot_model.fk_pos(self.state.q)
        )
        dq_current_norm = np.linalg.norm(self.state.dq)
        print(
            f"t = {self.state.timestep / 10:.1f}: Distance to goal {dist_goal:.3f}m, ||dq|| = {dq_current_norm:.3f}rad/s"
        )
        if dist_goal < 0.03 and dq_current_norm < 0.05:
            print("Reached goal")
            success = True

        self.metrics["success"] = success
        self.metrics["goal_distance"] = dist_goal
        self.metrics["velocity_norm"] = dq_current_norm
        # print(f"Flow time: {flow_time}")
        # print(f"Copy time: {copy_time}")
        # print(f"Sets time: {self.safety_filter.t_sets}")
        # print(f"Opt time: {self.metrics['opt_times'][-1]}")
        # print(f"Opt loop: {self.safety_filter.t_loop}")
        # print(time.perf_counter() - t_start)
        self.safety_filter.t_loop = 0.0
        if self.config.experiment:
            t_sleep = (
                self.robot_controller.t_current
                - self.robot_controller.iiwa.time.to_sec()
                - self.config.dt_sim
                - 0.015
            )
            time.sleep(max(0, t_sleep))
        elif self.config.sleep:
            time.sleep(
                max(0, self.config.dt_sim - 0.01 - (time.perf_counter() - t_start))
            )

    def _compute_guidance(self, x_current, dx_flow, q_start, p_goal):
        q_des = (
            (x_current + dx_flow)
            .reshape((-1, self.config.n_out))
            .detach()
            .cpu()
            .numpy()
        )
        if self.config.use_guidance:
            p_init, _, _ = self.robot_model.forward_kinematics(q_start, 0 * q_start)
            q_des_old = (
                x_current.reshape((-1, self.config.n_out)).detach().cpu().numpy()
            )
            for i in range(1, self.config.n_horizon):
                pc, jac, _ = self.robot_model.forward_kinematics(
                    q_des[i, :7], 0 * q_des[i, :7]
                )
                p_des = np.copy(p_goal[:3])

                # Set z coordinate to be the current one which effectively projects
                # the current position on the vertical axis at the goal pose
                # p_des[2] = max(p_des[2], pc[2])
                p_des[2] = pc[2]

                d0 = np.linalg.norm(p_des - p_init[:3])
                dc = np.linalg.norm(p_des - pc[:3])
                if dc <= 0.1:
                    d0 = np.linalg.norm(p_goal[:3] - p_init[:3])
                    dc = np.linalg.norm(p_goal[:3] - pc[:3])
                # scal = max(0, 1 - dc / d0) ** 2
                scal = max(0, np.exp(-10 * (dc / d0 - 0.09)))
                # horizon_scal = (i + 2) / self.config.n_horizon
                # horizon_scal = min(1, horizon_scal)
                # scal = max(0, np.exp(-50 / horizon_scal * (dc / d0 - 0.01)))
                scal = min(1, scal)
                dpose = np.zeros(6)
                dpose[:3] = p_des - pc[:3]

                # Use rotation error for orientation guidance
                r_current = R.from_rotvec(pc[3:])
                r_goal = R.from_rotvec(p_goal[3:])
                dR = r_current * r_goal.inv()
                dpose[3:] = -dR.as_rotvec()
                alpha = 0.001  # Regularization parameter
                jac_pinv = (
                    np.linalg.inv(jac.T @ jac + alpha * np.eye(jac.shape[1])) @ jac.T
                )
                # dq_des = jac_pinv @ dpose + 0.01 * (np.eye(7) - jac_pinv @ jac) @ (
                #     -q_des_old[i, :7]
                # )
                dq_des = jac_pinv @ dpose
                # params = np.concatenate((jac.T.flatten(), dpose))
                # sol = solver(
                #     x0=np.zeros(7),
                #     lbx=lbu,
                #     ubx=ubu,
                #     lbg=lbg,
                #     ubg=ubg,
                #     p=params,
                # )
                # dq_des = sol["x"].full().flatten()

                guidance_weight = scal * (i + 1) / self.config.n_horizon
                # guidance_weight = scal * 1.0
                # q_des[i, :] += guidance_weight * jac_pinv @ dpose
                # q_des[i, :] += guidance_weight * sol["x"].full().flatten()
                # print(guidance_weight)
                q_des[i, :7] = (1 - guidance_weight) * q_des[
                    i, :7
                ] + guidance_weight * (q_des_old[i, :7] + dq_des)
        return q_des[:, :7]

    def _update_state_and_metrics(
        self,
        loop_time: float,
        flow_time: float,
        safety_intervention: float,
        x_steps_current: List[np.ndarray],
    ) -> None:
        """Update simulation state and collect metrics."""
        x_np = (
            self.x_current.detach()
            .cpu()
            .numpy()
            .reshape((self.config.n_horizon, self.config.n_out))
        )

        if self.config.handover:
            if x_np[0, 7] < 0.1 and self.t_gripper_open == np.inf:
                print("Opening Gripper")
                self.t_gripper_open = self.state.timestep
                # print(
                #     np.linalg.norm(x_np[0, 8:11] - self.robot_model.fk_pos(x_np[0, :7]))
                # )
            # p_human = x_np[:, 8:11]
            # self.safety_filter.set_handover_data(p_human)

        # Update state
        next_q = x_np[self.config.n_actions, :7]
        self.state.timestep += 1
        if self.config.use_safety_filter:
            if not self.config.real_time:
                self.metrics["opt_times"].append(np.sum(self.safety_filter.t_array))
            self.safety_filter.update_initial_state(self.config.n_actions)
            self.safety_filter_init.q_last = self.safety_filter.q_last[:, :-1]
            self.safety_filter_init.dq_last = self.safety_filter.dq_last[:, :-1]
            self.safety_filter_init.ddq_last = self.safety_filter.ddq_last[:, :-1]
            self.safety_filter_init.dddq_last = self.safety_filter.dddq_last[:, :-1]
            self.safety_filter_init.update_initial_state(self.config.n_actions)
            self.state.q = self.safety_filter.q.flatten()
            self.state.dq = self.safety_filter.dq.flatten()
            self.state.ddq = self.safety_filter.ddq.flatten()
            self.state.dddq = self.safety_filter.dddq.flatten()
        else:
            self.state.q = next_q

        # Update history
        for k in range(self.config.n_actions):
            self.state.q_prev = self.state.q_prev[1:]
            self.state.q_prev.append(x_np[k + 1, :7])

        if self.config.use_safety_filter and not self.config.real_time:
            # Store derivative trajectories for each action step
            for k in range(self.config.n_actions):
                dx_k = np.concatenate(
                    (self.safety_filter.dq_last.T[k:], np.zeros((k, 7)))
                )
                ddx_k = np.concatenate(
                    (self.safety_filter.ddq_last.T[k:], np.zeros((k, 7)))
                )
                dddx_k = np.concatenate(
                    (self.safety_filter.dddq_last.T[k:], np.zeros((k, 7)))
                )

                self.metrics["dx_plot"].append(dx_k)
                self.metrics["ddx_plot"].append(ddx_k)
                self.metrics["dddx_plot"].append(dddx_k)

        if not self.config.real_time:
            # Store all flow steps for this timestep
            self.metrics["x_steps"].append(x_steps_current)

            # Store trajectory data for multiple action steps
            for k in range(self.config.n_actions):
                # Shift trajectory for multi-step planning
                x_k = np.concatenate((x_np[k:, :7], np.zeros((k, 7))))
                self.metrics["x_plot"].append(x_k)

            # Collect metrics
            pc, _, _ = self.robot_model.forward_kinematics(
                self.state.q, 0 * self.state.q
            )
            self.metrics["positions"].append(pc)

            # Store timing metrics for each action step
            for k in range(self.config.n_actions):
                self.metrics["times"].append(loop_time)
                self.metrics["flow_times"].append(flow_time)
                self.metrics["safety_interventions"].append(safety_intervention)

    def _update_visualization(
        self,
    ) -> None:
        """Update RViz visualization."""
        if not self.config.real_time:
            x_cj_np = (
                self.x_current.detach()
                .cpu()
                .numpy()
                .reshape((self.config.n_horizon, self.config.n_out))
            )

            # Current and goal positions
            pc, _, _ = self.robot_model.forward_kinematics(
                self.state.q, 0 * self.state.q
            )

            # Trajectory path
            p_traj = np.empty((6, self.config.n_horizon))
            for i in range(x_cj_np.shape[0]):
                p_traj[:, i], _, _ = self.robot_model.forward_kinematics(
                    x_cj_np[i, :7], 0 * self.state.q
                )

            self.data.qpos[:7] = self.state.q
            mujoco.mj_forward(self.model, self.data)
            # Sync the viewer with the updated simulation data
            self.viewer.sync()

    def calculate_collision_distances(self, state: SimulationState) -> None:
        """Calculate minimum distances to obstacles for all joints."""
        for k in range(self.config.n_actions):
            qc = state.q if k == 0 else state.q  # Use current state for all actions

            # Get collision sphere positions for all joints
            nr_p_col = len(self.robot_model.col_names)
            p_list = [self.robot_model.fk_pos_col(qc, i) for i in range(nr_p_col)]

            d_p = np.zeros(nr_p_col)
            for i, p in enumerate(p_list):
                d_all_obs = []

                # Check distance to all obstacles
                obs_sets = self.safety_filter.set_finder.obs_sets

                for obs in obs_sets:
                    # Check if point is outside obstacle
                    do = obs[0] @ p - obs[1]
                    if np.max(do) >= -1e-6:
                        # Point is outside, solve for closest point on obstacle
                        params = np.concatenate((obs[0].T.flatten(), obs[1], p))
                        sol = self.collision_solver(
                            x0=p,
                            lbx=-np.inf * np.ones(3),
                            ubx=np.inf * np.ones(3),
                            lbg=self.lbg,
                            ubg=self.ubg,
                            p=params,
                        )
                        distance = (
                            np.linalg.norm(sol["x"].full().flatten() - p)
                            - self.robot_model.col_joint_sizes[i]
                        )
                        d_all_obs.append(distance)
                    else:
                        # Point is inside obstacle (negative distance)
                        d_all_obs.append(np.max(do))
                d_p[i] = np.min(d_all_obs)

            self.metrics["collision_distances"].append(d_p)

    def get_trajectory_arrays(self) -> Dict[str, np.ndarray]:
        """Convert trajectory metrics to numpy arrays for analysis."""
        arrays = {}

        arrays["x_plot"] = np.array(self.metrics["x_plot"])
        arrays["positions"] = np.array(self.metrics["positions"])
        arrays["times"] = np.array(self.metrics["times"])
        arrays["flow_times"] = np.array(self.metrics["flow_times"])
        arrays["safety_interventions"] = np.array(self.metrics["safety_interventions"])

        arrays["dx_plot"] = np.array(self.metrics["dx_plot"])
        arrays["ddx_plot"] = np.array(self.metrics["ddx_plot"])
        arrays["dddx_plot"] = np.array(self.metrics["dddx_plot"])
        arrays["opt_times"] = np.array(self.metrics["opt_times"])

        arrays["x_steps"] = self.metrics["x_steps"]
        arrays["collision_distances"] = np.array(self.metrics["collision_distances"]).T

        return arrays

    def plot_results(self, t_ref=None, q_ref=None) -> None:
        """Generate analysis plots of the planning results."""
        arrays = self.get_trajectory_arrays()

        # Joint trajectory plots
        plt.figure(figsize=(12, 8))
        x_plot = arrays["x_plot"]
        t = np.arange(
            0.0, x_plot.shape[0] * self.config.dt_sim - 0.05, self.config.dt_sim
        )

        for k in range(7):
            plt.subplot(2, 4, k + 1)
            q = x_plot[:, 0, k]
            plt.plot(t, q, f"C{k}", label=f"Joint {k}")
            if t_ref is not None and q_ref is not None:
                plt.plot(t_ref, q_ref[:, k], f"C{k}--", label=f"Joint Ref {k}")
            plt.title(f"Joint {k} Position")
            plt.xlabel("Time (s)")
            plt.ylabel("Position (rad)")
            plt.hlines(
                self.robot_model.q_lim_upper[k],
                t[0],
                t[-1],
                colors="r",
                linestyles="--",
            )
            plt.hlines(
                self.robot_model.q_lim_lower[k],
                t[0],
                t[-1],
                colors="r",
                linestyles="--",
            )
            for i in range(0, x_plot.shape[0], self.config.n_actions):
                t_horizon = np.arange(
                    i * 0.1, (i + self.config.n_horizon) * 0.1 - 0.05, 0.1
                )
                q_horizon = x_plot[i, :, k]
                plt.plot(t_horizon, q_horizon, f"C{k}", label="_none_", linewidth=0.2)
            plt.legend()

        plt.tight_layout()
        plt.suptitle("Joint Trajectories")

        # Derivative plots if available
        if "dx_plot" in arrays:
            plt.figure(figsize=(15, 10))
            dx_plot = arrays["dx_plot"]
            ddx_plot = arrays["ddx_plot"]
            dddx_plot = arrays["dddx_plot"]

            for k in range(7):
                # Velocities
                plt.subplot(3, 7, k + 1)
                dq = dx_plot[:, 0, k]
                plt.plot(t, dq, f"C{k}")
                plt.hlines(
                    self.robot_model.dq_lim_lower[k],
                    t[0],
                    t[-1],
                    colors="r",
                    linestyles="--",
                )
                plt.hlines(
                    self.robot_model.dq_lim_upper[k],
                    t[0],
                    t[-1],
                    colors="r",
                    linestyles="--",
                )
                plt.title(f"Joint {k} Velocity")

                # Accelerations
                plt.subplot(3, 7, k + 8)
                ddq = ddx_plot[:, 0, k]
                plt.plot(t, ddq, f"C{k}")
                plt.hlines([-5.0, 5.0], t[0], t[-1], colors="r", linestyles="--")
                plt.title(f"Joint {k} Acceleration")

                # Jerks
                plt.subplot(3, 7, k + 15)
                dddq = dddx_plot[:, 0, k]
                plt.plot(t, dddq, f"C{k}")
                plt.hlines(
                    [self.robot_model.u_min, self.robot_model.u_max],
                    t[0],
                    t[-1],
                    colors="r",
                    linestyles="--",
                )
                plt.title(f"Joint {k} Jerk")

            plt.tight_layout()
            plt.suptitle("Joint Derivatives")

        # Timing analysis
        plt.figure(figsize=(10, 6))
        plt.plot(t, arrays["times"], "C0", label="Loop Time")
        if "opt_times" in arrays:
            t_opt = np.arange(
                0.0,
                arrays["opt_times"].shape[0]
                * (self.config.dt_sim * self.config.n_actions)
                - 0.05,
                self.config.dt_sim * self.config.n_actions,
            )
            plt.plot(
                t_opt,
                arrays["opt_times"],
                "C1--",
                label="Safety Filter Total",
            )

        plt.plot(t, arrays["flow_times"], "C2--", label="Flow Network Total")
        plt.hlines(
            self.config.dt_sim,
            t[0],
            t[-1],
            colors="r",
            linestyles="-",
            label="Real-time Limit",
        )
        plt.xlabel("Time (s)")
        plt.ylabel("Computation Time (s)")
        plt.title("Computation Times")
        plt.legend()

        # Flow matching step visualization
        plt.figure(figsize=(15, 8))
        skip_plots = 2
        n_timesteps_to_show = len(arrays["x_steps"])

        for i in range(0, n_timesteps_to_show, skip_plots):
            x_sc = np.array(arrays["x_steps"][i])
            x_sc = x_sc.reshape(
                (x_sc.shape[0], self.config.n_horizon, self.config.n_out)
            )
            x_sc = x_sc[:, :, :7]

            tc = np.linspace(
                0.1 * i, 0.1 * i + (self.config.n_horizon - 1) / 10, x_sc.shape[1]
            )

            # Plot each flow step
            for flow_step in range(0, x_sc.shape[0], skip_plots):
                subplot_idx = int(flow_step / skip_plots) + 1
                plt.subplot(
                    1, int(self.config.flow_steps / skip_plots) + 1, subplot_idx
                )

                q_traj = x_sc[flow_step, :, :]
                alpha = 0.3 + 0.7 * (
                    flow_step / (x_sc.shape[0] - 1)
                )  # Fade from light to dark

                for j in range(7):
                    plt.plot(tc, q_traj[:, j], f"C{j}", alpha=alpha, linewidth=0.5)
                plt.ylim([-np.pi, np.pi])

            plt.xlabel("Time (s)")
            plt.ylabel("Joint Position (rad)")

        plt.suptitle("Flow Matching Evolution")
        plt.tight_layout()

        # Collision distance plots
        plt.figure(figsize=(12, 6))
        collision_distances = arrays["collision_distances"]
        joint_names = getattr(
            self.robot_model, "col_names", [f"Joint_{i}" for i in range(7)]
        )

        # Determine actual data length
        n_data_points = np.sum(np.any(collision_distances != 0, axis=0))
        if n_data_points == 0:
            n_data_points = (
                len(arrays["times"])
                if "times" in arrays
                else collision_distances.shape[1]
            )

        t_collision = np.arange(
            0.0, n_data_points * self.config.dt_sim - 0.05, self.config.dt_sim
        )

        plt.subplot(2, 1, 1)
        for j in range(collision_distances.shape[0]):
            if j < len(joint_names):
                label = joint_names[j]
            else:
                label = f"Joint_{j}"
            plt.plot(
                t_collision,
                collision_distances[j, : len(t_collision)],
                f"C{j}",
                label=label,
            )

        plt.hlines(
            [0.0],
            t_collision[0],
            t_collision[-1],
            colors="r",
            linestyles="-",
            label="Collision Boundary",
        )
        plt.legend()
        plt.title("Safety Set Containment / Collision Distances")
        plt.xlabel("Time (s)")
        plt.ylabel("Distance to Obstacle (m)")

        plt.subplot(2, 1, 2)
        plt.plot(arrays["safety_interventions"][: len(t_collision)])
        plt.title("Safety Interventions")
        plt.xlabel("Time (s)")
        plt.ylabel("Intervention Magnitude")

        plt.tight_layout()

        plt.show()
