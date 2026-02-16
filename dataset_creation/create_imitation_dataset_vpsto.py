from glob import glob
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R

from safe_flow_mpc.RobotModel import RobotModel
from safe_flow_mpc.SafeFlowMPC.ObstacleManager import ObstacleManager
from safe_flow_mpc.SafetyFilter import (
    SafetyFilter,
)

robot_model = RobotModel()
rng = np.random.default_rng()


def sample_initial_pos(shape, q0, dq0, ddq0, dddq0):
    jerk_0 = 35 * (2 * (rng.random(shape) - 0.5))
    d_q0 = 0.3 * (2 * (rng.random(7) - 0.5)) * 0
    d_dq0 = 10 * (2 * (rng.random(7) - 0.5)) * 0
    d_ddq0 = 10 * (2 * (rng.random(7) - 0.5)) * 0
    d_dddq0 = 35 * (2 * (rng.random(7) - 0.5)) * 0
    pos_0 = jerk_to_pos(
        jerk_0, q0=q0 + d_q0, dq0=dq0 + d_dq0, ddq0=ddq0 + d_ddq0, dddq0=dddq0 + d_dddq0
    )
    return pos_0


def jerk_to_pos(u, q0, dq0, ddq0, dddq0, dt=0.1):
    u = u.reshape((u.shape[0], -1, 7))
    q = np.zeros((u.shape[0], u.shape[1], u.shape[2]))
    dq = np.zeros_like(q)
    ddq = np.zeros_like(q)
    q[:, 0, :] = q0
    dq[:, 0, :] = dq0
    ddq[:, 0, :] = ddq0
    u[:, 0, :] = dddq0

    for k in range(q.shape[1] - 1):
        q[:, k + 1, :] = (
            ddq[:, k, :] * dt**2 / 2.0
            + dq[:, k, :] * dt
            + q[:, k, :]
            + u[:, k, :] * dt**3 / 8.0
            + u[:, k + 1, :] * dt**3 / 24.0
        )
        dq[:, k + 1, :] = (
            ddq[:, k, :] * dt
            + dq[:, k, :]
            + u[:, k, :] * dt**2 / 3.0
            + u[:, k + 1, :] * dt**2 / 6.0
        )
        ddq[:, k + 1, :] = (
            ddq[:, k, :] + u[:, k, :] * dt / 2.0 + u[:, k + 1, :] * dt / 2.0
        )

    # q = q[:, 1:, :]
    return q


def process_file(i):
    n_samples = 10
    n_horizon = 16
    n_steps = 50
    trajs = []
    t_samples = []
    samples = []
    dsamples = []
    c_data1 = []
    c_data2 = []
    c_data3 = []
    print(f"Traj: {i}")
    data = np.load(path + f"traj_vpsto_{i}.npz")
    obstacle_manager = ObstacleManager()
    obstacle_manager.add_default_obstacles()
    safety_filter = SafetyFilter(
        N=n_horizon,
        smooth=False,
        use_term=use_term,
        use_sets=use_sets,
        obstacle_manager=obstacle_manager,
    )
    for j in range(data["q"].shape[0]):
        # print(f"Traj: {i}, Step {j}")
        # traj = data["q"][j, :, :].T.flatten()
        if j < data["q"].shape[0] - (n_horizon - 1):
            traj = data["q"][j : j + n_horizon, :].flatten()
        else:
            traj0 = data["q"][j:, :].flatten()
            remaining_length = n_horizon - (data["q"].shape[0] - j)
            traj1 = (np.ones((remaining_length, 7)) * data["q"][-1, :]).flatten()
            traj = np.concatenate((traj0, traj1))
        if j <= 9:
            idx_init = 10 - j
            idx_prev = [j - x for x in reversed(range(10 - idx_init))]
            q_prev1 = data["q_prev0"][-idx_init:, :]
            q_prev2 = data["q"][idx_prev, :]
            q_prev = np.vstack((q_prev1, q_prev2)).flatten()
        else:
            idx_prev = [j - x for x in reversed(range(10))]
            q_prev = data["q"][idx_prev, :].flatten()
        q0 = data["q"][j, :]
        q_final = data["q"][-1, :]
        dq0 = data["dq"][j, :]
        ddq0 = data["ddq"][j, :]

        h = robot_model.hom_transform_endeffector(q0)
        p0 = h[:3, 3]
        r0 = h[:3, :3].flatten()
        h = robot_model.hom_transform_endeffector(q_final)
        p_final = h[:3, 3]
        r_final = h[:3, :3].flatten()

        p_cols = [robot_model.fk_pos_col(q0, i) for i in range(7)]

        conditional_data1 = np.concatenate([q_prev, p0, r0, p_final, r_final] + p_cols)
        conditional_data2 = np.concatenate([q_prev, p0, r0, p_final, r_final])
        conditional_data3 = np.concatenate((q_prev, q_final))

        # Safe sampling
        dddq0 = np.zeros(7)  # TODO improve this
        fail = False
        try:
            samples_new = []
            dsamples_new = []
            t_samples_new = []
            for k in range(n_samples):
                safety_filter.reset()
                safety_filter.updated = True
                sample = sample_initial_pos((1, n_horizon, 7), q0, dq0, ddq0, dddq0)
                sample = sample[0, :, :]
                # Make the sample safe
                sample = safety_filter.step(
                    q0,
                    sample,
                    dq0=dq0,
                    ddq0=ddq0,
                    dddq0=dddq0,
                    # u_init=data["dddq"][j, :, :].T,
                ).T
                q_f = np.copy(traj.reshape((n_horizon, 7)))[:n_horizon, :]
                q_f_orig = np.copy(q_f)
                if use_term:
                    # Make the desired trajectory safe
                    q_f = safety_filter.step(
                        q_f[0, :],
                        q_f,
                        dq0=dq0,
                        ddq0=ddq0,
                        dddq0=dddq0,
                    ).T
                q_des = q_f
                samplesk = []
                t_samplesk = []
                for l in range(n_steps + 1):
                    t = l * 1 / n_steps
                    # w = 1 - t**6
                    # t_adjusted = w * 1 / n_steps + (1 - w) * t
                    # q_des = q_des + (sample - q_des) * t_adjusted
                    q_des = q_f + (sample - q_f) * t
                    q_t = safety_filter.step(
                        q0,
                        q_des,
                        dq0=dq0,
                        ddq0=ddq0,
                        dddq0=dddq0,
                    )
                    q_des = np.copy(q_t.T)
                    samplesk.append(q_t.copy())
                    t_samplesk.append(t)
                t_samples_new.append(t_samplesk[:-1])
                samples_new.append(list(reversed(samplesk)))
                dx_des = np.diff(np.array(list(reversed(samplesk))), axis=0) / (
                    1 / n_steps
                )
                dsamples_new.append(list(dx_des))
                # (sample[:, 1] - q_f[:, 1]) + np.cumsum(1 / n_steps * np.array(dsamples[-1][-1]), axis=0)[-1, 1, :]
                # samples_new = list(reversed(samples_new))

                # plt.figure()
                # for i in range(n_steps):
                #     for l in range(7):
                #         plt.subplot(2, 4, l + 1)
                #         lw = 0.1 + i / 100
                #         plt.plot(samplesk[i][l, :], "C0", linewidth=lw)
                #         if i == 0:
                #             plt.plot(q_f[:, l], "C1", linewidth=1, label="q_f")
                #             plt.plot(
                #                 q_f_orig[:, l], "C1--", linewidth=1, label="q_f_orig"
                #             )
                #             plt.plot(sample[:, l], "C2", linewidth=1, label="Sample")
                #             if l == 0:
                #                 plt.legend()
                # plt.suptitle(f"Time {j / 10}")
                # # plt.show()
                # plt.figure()
                # for l in range(7):
                #     for i in range(n_steps):
                #         plt.subplot(2, 4, l + 1)
                #         plt.plot(i, samplesk[i][l, -1], "C0.")
                #         if i == n_steps - 1:
                #             plt.plot(
                #                 [0, n_steps], [q_f[-1, l]] * 2, "C1--", label="q_f"
                #             )
                #             plt.plot(
                #                 [0, n_steps],
                #                 [sample[-1, l]] * 2,
                #                 "C2--",
                #                 label="Sample",
                #             )
                #         if i == n_steps - 1 and l == 0:
                #             plt.legend()
                # plt.suptitle(f"Time {j / 10}")
                # plt.show()

            # plt.figure()
            # for i in range(7):
            #     plt.plot(range(10), q_prev.reshape((-1, 7))[:, i], f"C{i}--")
            #     plt.plot(range(9, 25), traj.reshape((-1, 7))[:, i], f"C{i}--")
            # plt.show()

        # safety_filter.update_initial_state()
        except RuntimeError as e:
            print(f"[Warning] File {i} time {j} failed")
            fail = True

        if not fail:
            trajs.append(traj)
            c_data1.append(conditional_data1)
            c_data2.append(conditional_data2)
            c_data3.append(conditional_data3)
            t_samples.append(t_samples_new)
            dsamples.append(dsamples_new)
            samples.append(samples_new)
    return (
        trajs,
        c_data1,
        c_data2,
        c_data3,
        t_samples,
        samples,
        dsamples,
    )


path = "data/"
use_term = True
use_sets = False
n_file_batch = 20
min_file_idx = 0
max_file_idx = len(glob(f"{path}traj_vpsto_*.npz"))
if max_file_idx < n_file_batch:
    n_file_batch = max_file_idx
for i in range(int(min_file_idx / n_file_batch), int(max_file_idx / n_file_batch)):
    file_idxs = range(i * n_file_batch, (i + 1) * n_file_batch, 1)
    with Pool(processes=20) as pool:  # Use number of CPU cores or less
        results = pool.map(process_file, file_idxs)

    traj_array = []
    c_array1 = []
    c_array2 = []
    c_array3 = []
    t_samples_array = []
    samples_array = []
    dsamples_array = []
    for result in results:
        if result[0] != []:
            traj_array += result[0]
            c_array1 += result[1]
            c_array2 += result[2]
            c_array3 += result[3]
            t_samples_array += result[4]
            samples_array += result[5]
            dsamples_array += result[6]
    save_data = {}
    save_data["trajectories"] = np.array(traj_array)
    save_data["c_data1"] = np.array(c_array1)
    save_data["c_data2"] = np.array(c_array2)
    save_data["c_data3"] = np.array(c_array3)
    save_data["samples"] = np.array(samples_array)
    save_data["dsamples"] = np.array(dsamples_array)
    save_data["t_samples"] = np.array(t_samples_array)
    print(f"Saving batch {i}")
    np.savez(
        path + f"imitation_trajs_vpsto{'_term' if use_term else ''}_{i}.npz",
        **save_data,
    )
