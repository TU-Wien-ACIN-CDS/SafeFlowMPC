from glob import glob

import numpy as np

from safe_flow_mpc.RobotModel import RobotModel

"""
Create a dataset in the correct format to make it easier to load it for training.
"""

robot_model = RobotModel()
path = "data/"
traj_array = []
traj_last_array = []
c_array1 = []
c_array2 = []
c_array3 = []
l_longest = 0
n_horizon = 16
for i, fpath in enumerate(glob(f"{path}traj_vpsto_*.npz")):
    trajs = []
    trajs_last = []
    c_data1 = []
    c_data2 = []
    c_data3 = []
    print(f"Traj: {i}")
    try:
        data = np.load(fpath)
        if data["q"].shape[0] > l_longest and data["q"].shape[0] < 100:
            l_longest = data["q"].shape[0]
        for j in range(data["q"].shape[0]):
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

            idx_last = [j - 1 + x for x in range(n_horizon)]
            idx_last = np.maximum(idx_last, 0)
            idx_last = np.minimum(idx_last, data["q"].shape[0] - 1)
            traj_last = data["q"][idx_last, :].flatten()

            h = robot_model.hom_transform_endeffector(q0)
            p0 = h[:3, 3]
            r0 = h[:3, :3].flatten()
            h = robot_model.hom_transform_endeffector(q_final)
            p_final = h[:3, 3]
            r_final = h[:3, :3].flatten()

            p_cols = [robot_model.fk_pos_col(q0, i) for i in range(7)]

            conditional_data1 = np.concatenate(
                [q_prev, p0, r0, p_final, r_final] + p_cols
            )
            conditional_data2 = np.concatenate([q_prev, p0, r0, p_final, r_final])
            conditional_data3 = np.concatenate((q_prev, q_final))

            trajs.append(traj)
            trajs_last.append(traj_last)
            c_data1.append(conditional_data1)
            c_data2.append(conditional_data2)
            c_data3.append(conditional_data3)
        traj_array += trajs
        traj_last_array += trajs_last
        c_array1 += c_data1
        c_array2 += c_data2
        c_array3 += c_data3
    except FileNotFoundError:
        print(f"[Warning] File {i} not found")

save_data = {}
save_data["trajectories"] = np.array(traj_array)
save_data["trajectories_last"] = np.array(traj_last_array)
save_data["c_data1"] = np.array(c_array1)
save_data["c_data2"] = np.array(c_array2)
save_data["c_data3"] = np.array(c_array3)
np.savez(path + "imitation_trajs_vpsto_unsafe.npz", **save_data)
