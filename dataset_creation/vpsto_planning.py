import argparse
import time

import casadi as ca
import mujoco
import mujoco.viewer
import numpy as np
from scipy.spatial.transform import Rotation as R
from vpsto.vpsto.vpsto import VPSTO, VPSTOOptions

from safe_flow_mpc.RobotModel import RobotModel

mj_model = mujoco.MjModel.from_binary_path("mujoco_model/mujoco_env.mjb")
data = mujoco.MjData(mj_model)
robot_model = RobotModel()


def eval_third_order_poly(x, a_list):
    al = [np.expand_dims(a_list[i], 1) for i in range(len(a_list))]
    y = al[0] + al[1] * x + al[2] * x**2 + al[3] * x**3
    dy = al[1] + 2 * al[2] * x + 3 * al[3] * x**2
    ddy = 2 * al[2] + 6 * al[3] * x
    return y, dy, ddy


def setup_ik_problem_with_obs():
    q = ca.SX.sym("q", 7)
    pd = ca.SX.sym("p desired", 3)
    rd = ca.SX.sym("r desired", 3, 3)
    r_ee = robot_model.hom_transform_endeffector(q)[:3, :3]
    J = ca.sumsqr(robot_model.fk_pos(q) - pd[:3])
    J += ca.sumsqr(r_ee @ rd.T - ca.SX.eye(3))
    w = [q]
    lbw = robot_model.q_lim_lower
    ubw = robot_model.q_lim_upper

    obs = ca.SX.sym("q", 4, 2)

    g = []
    lbg = []
    ubg = []
    p_list = [robot_model.fk_pos_col(q, i) for i in range(7)]
    for p in p_list:
        for i in range(obs.shape[1]):
            d = ca.sumsqr(p - obs[:3, i])
            g += [d - obs[3, i] ** 2]
            lbg += [0]
            ubg += [np.inf]
        # g += [p[2]]
        # lbg += [0]
        # ubg += [1.1]
        # g += [p[0]]
        # lbg += [-0.2]
        # ubg += [np.inf]

    params = ca.vertcat(pd, rd.reshape((-1, 1)), obs.reshape((-1, 1)))

    prob = {
        "f": J,
        "x": ca.vertcat(*w),
        "g": ca.vertcat(*g),
        "p": params,
    }
    lbu = lbw
    ubu = ubw
    ipopt_options = {
        "tol": 10e-4,
        "max_iter": 500,
        "print_info_string": "no",
        "print_level": 0,
    }

    solver_opts = {
        "verbose": False,
        "verbose_init": False,
        "print_time": False,
        "ipopt": ipopt_options,
    }
    ik_solver = ca.nlpsol("solver", "ipopt", prob, solver_opts)
    return ik_solver, lbu, ubu, lbg, ubg


def loss_limits(candidates):
    q_mins = robot_model.q_lim_lower
    q_maxs = robot_model.q_lim_upper

    q = candidates["pos"]
    d_min = np.maximum(np.zeros_like(q), -q + q_mins)
    d_max = np.maximum(np.zeros_like(q), q - q_maxs)
    return np.sum(d_min > 1e-6, axis=(1, 2)) + np.sum(d_max > 1e-6, axis=(1, 2))


def loss_collision(candidates, mj_model, mj_data):
    costs = []
    for traj in candidates["pos"]:
        cost_traj = 0
        for pos in traj:
            mj_data.qpos[:7] = pos
            mujoco.mj_forward(mj_model, mj_data)
            cost_traj += np.sum(np.abs(mj_data.contact.dist)) + len(
                mj_data.contact.dist
            )
        costs.append(cost_traj)
    costs = np.array(costs)
    costs += costs > 0.0
    return costs


def loss_target(candidates, mj_model, mj_data):
    qs = candidates["pos"]
    costs = []
    for q in qs[:, -1, :]:
        h = robot_model.hom_transform_endeffector(q)
        p = h[:3, 3]
        r = h[:3, :3]
        dp = np.linalg.norm(p - p1)
        dr = np.linalg.norm(R.from_matrix(r @ r1.T).as_rotvec())
        cost = dp + dr  # > 1e-3
        costs.append(cost)
    return np.array(costs)


def loss(candidates, mj_model, data):
    # cost_target = loss_target(candidates, mj_model, data)
    cost_collision = loss_collision(candidates, mj_model, data)
    cost_limits = loss_limits(candidates)
    # return (
    #     candidates["T"] + 1e1 * cost_target + 1e3 * cost_collision + 1e3 * cost_limits
    # )
    return candidates["T"] + 1e3 * cost_collision + 1e3 * cost_limits


def main():
    parser = argparse.ArgumentParser(description="Process trajectories.")
    parser.add_argument(
        "--nr_trajs",
        type=int,
        required=True,
        default=4000,
        help="Number of trajectories to process.",
    )
    args = parser.parse_args()

    opt = VPSTOOptions(ndof=7)
    opt.vel_lim = robot_model.dq_lim_upper
    opt.acc_lim = 4.0 * np.ones(7)
    opt.N_via = 8
    opt.N_eval = 100
    opt.pop_size = 10
    opt.max_iter = 100
    opt.sigma_init = 0.5

    ik_solver, lbu, ubu, lbg, ubg = setup_ik_problem_with_obs()
    obs = np.zeros((4, 2))
    obs[:3, 0] = np.array([0.65, 0.0, 0.8])
    obs[3, 0] = 0.075
    obs[:3, 1] = np.array([0.65, 0.0, 0.4])
    obs[3, 1] = 0.075

    traj_opt = VPSTO(opt)

    path = "data/"
    rng = np.random.default_rng()
    q_last = None
    dq_last = None
    t_best_last = None
    saved_idx = 0
    nr_trajs = args.nr_trajs
    print(f"Number of trajectories: {nr_trajs}")
    create_same_side_adjust = False
    create_stuck_help = False
    nr_normal_traj = nr_trajs // 2
    nr_same_side_adjust = nr_trajs // 4
    while saved_idx <= nr_trajs:
        if saved_idx >= nr_normal_traj:
            create_same_side_adjust = True
        if saved_idx >= nr_normal_traj + nr_same_side_adjust:
            create_stuck_help = True

        if create_same_side_adjust or create_stuck_help:
            new_traj = True
        else:
            new_traj = True if saved_idx % 2 or q_last is None else False

        # Find initial configuration and pose
        if new_traj:
            valid_ik = False
            initial_collision = True
            while valid_ik is not True or initial_collision:
                p0 = rng.uniform(
                    [0.25, -0.6, 0.1],
                    [0.65, 0.6, 0.6],
                    3,
                )
                if create_stuck_help:
                    p0 = rng.uniform(
                        [0.4, -0.3, 0.1],
                        [0.7, 0.3, 0.4],
                        3,
                    )
                rand_rz = rng.uniform(-np.pi / 2, np.pi / 2, 2)
                r0 = R.from_euler(
                    "xyz", [rand_rz[1], np.abs(rand_rz[0]), 0]
                ).as_matrix()
                if create_stuck_help:
                    r0 = R.from_euler(
                        "xyz", [rand_rz[1], np.pi / 2 + 0.25 * rand_rz[0], 0]
                    ).as_matrix()
                if create_same_side_adjust:
                    p0[2] = rng.uniform([0.14], [0.25], 1)[0]
                    rand_z = rng.uniform([-np.pi / 2], [np.pi / 2], 1)[0]
                    r0 = R.from_euler("zyz", [0, np.pi / 2, rand_z]).as_matrix()
                params = np.concatenate((p0, r0.T.flatten(), obs.T.flatten()))
                sol = ik_solver(
                    x0=rng.uniform(robot_model.q_lim_lower, robot_model.q_lim_upper, 7),
                    lbx=lbu,
                    ubx=ubu,
                    lbg=lbg,
                    ubg=ubg,
                    p=params,
                )
                q0 = np.array(sol["x"]).flatten()
                if not ik_solver.stats()["success"]:
                    print("(IK) ERROR No convergence in IK optimization")
                h_ik = robot_model.hom_transform_endeffector(q0)
                pos_error = np.linalg.norm(p0 - h_ik[:3, 3])
                rot_error = np.linalg.norm(
                    R.from_matrix(h_ik[:3, :3] @ r0.T).as_rotvec()
                )
                print(f"(IK) Position error {pos_error}m")
                print(f"(IK) Rotation error {rot_error * 180 / np.pi} deg")
                if pos_error < 0.2 and rot_error < 45 * np.pi / 180:
                    valid_ik = True

                data.qpos[:7] = q0
                mujoco.mj_forward(mj_model, data)
                cost_traj = np.sum(np.abs(data.contact.dist)) + len(data.contact.dist)
                initial_collision = True if cost_traj > 1e-3 else False
            dq0 = np.zeros_like(q0)
            q_prev0 = np.vstack([q0] * 10)
        else:
            idx = rng.integers(10, int(10 * t_best_last) - 10, 1)[0]
            q0 = q_last[idx, :]
            dq0 = dq_last[idx, :]
            idx_prev = [idx - x for x in reversed(range(10))]
            idx_prev = np.maximum(idx_prev, 0)
            q_prev0 = q_last[idx_prev, :]
        h_ik = robot_model.hom_transform_endeffector(q0)
        p0 = h_ik[:3, 3]
        r0 = h_ik[:3, :3]

        # Find end configuration and pose
        valid_ik = False
        max_tries = 30
        tries = 0
        while valid_ik is not True:
            if p0[1] < 0.0:
                p1 = rng.uniform(
                    [0.25, 0.2, 0.25],
                    [0.8, 0.8, 0.25],
                    3,
                )
            else:
                p1 = rng.uniform(
                    [0.25, -0.8, 0.25],
                    [0.8, -0.2, 0.25],
                    3,
                )
            rand_z = rng.uniform([-np.pi / 2], [np.pi / 2], 1)[0]
            r1 = R.from_euler("zyz", [0, np.pi / 2, rand_z]).as_matrix()
            if create_same_side_adjust:
                if p0[1] > 0.0:
                    p1 = rng.uniform(
                        [0.25, 0.3, 0.25],
                        [0.8, 0.8, 0.25],
                        3,
                    )
                else:
                    p1 = rng.uniform(
                        [0.25, -0.8, 0.25],
                        [0.8, -0.3, 0.25],
                        3,
                    )
                rand_z = rng.uniform([-np.pi / 2], [np.pi / 2], 1)[0]
                r1 = R.from_euler("zyz", [0, np.pi / 2, rand_z]).as_matrix()
            # q1, pos_error, rot_error = robot_model.inverse_kinematics(p1, r1, q0)
            params = np.concatenate((p1, r1.T.flatten(), obs.T.flatten()))
            sol = ik_solver(x0=q0, lbx=lbu, ubx=ubu, lbg=lbg, ubg=ubg, p=params)
            q1 = np.array(sol["x"]).flatten()
            if not ik_solver.stats()["success"]:
                print("(IK) ERROR No convergence in IK optimization")
            h_ik = robot_model.hom_transform_endeffector(q1)
            pos_error = np.linalg.norm(p1 - h_ik[:3, 3])
            rot_error = np.linalg.norm(R.from_matrix(h_ik[:3, :3] @ r1.T).as_rotvec())
            print(f"(IK) Position error {pos_error}m")
            print(f"(IK) Rotation error {rot_error * 180 / np.pi} deg")
            data.qpos[:7] = q1
            mujoco.mj_forward(mj_model, data)
            cost_traj = np.sum(np.abs(data.contact.dist))
            initial_collision = True if cost_traj > 1e-3 else False

            if (
                pos_error < 0.01
                and rot_error < 5 * np.pi / 180
                and not initial_collision
            ):
                # Check the grasp position
                p_grasp = p1 - np.array([0, 0, 0.1])
                q_grasp, pe_grasp, re_grasp = robot_model.inverse_kinematics(
                    p_grasp, r1, q1
                )
                data.qpos[:7] = q_grasp
                mujoco.mj_forward(mj_model, data)

                if pe_grasp < 0.01 and re_grasp < 5 * np.pi / 180:
                    valid_ik = True
            else:
                print(f"(IK) Position error {pos_error}m")
                print(f"(IK) Rotation error {rot_error * 180 / np.pi} deg")
                print("(IK) ERROR No convergence in IK optimization")
            tries += 1
            if tries >= max_tries:
                print("No valid IK solution found, aborting....")
                break
        h_ik = robot_model.hom_transform_endeffector(q1)
        p1 = h_ik[:3, 3]
        r1 = h_ik[:3, :3]
        h_ik = robot_model.hom_transform_endeffector(q_grasp)
        p_grasp = h_ik[:3, 3]
        r_grasp = h_ik[:3, :3]

        if valid_ik:
            print()
            print("------------------")
            print(f"Planning {saved_idx}:")
            print(f"q0: {q0}")
            print(f"dq0: {dq0}")
            print(f"p0: {p0}")
            print(f"r0: {r0}")
            print(f"p1: {p1}")
            print(f"r1: {r1}")
            print("------------------")
            print()

            # Compute grasp trajectory first
            jac = robot_model.jacobian_fk(q1)
            inv_jac = np.linalg.pinv(jac)
            dq_des = inv_jac @ np.array([0, 0, -1, 0, 0, 0]) * 0.1
            a_list = [
                q1,
                dq_des,
                3 * q_grasp - 3 * q1 - 2 * dq_des,
                -2 * q_grasp + 2 * q1 + dq_des,
            ]

            # dqT = np.zeros_like(q0)
            dqT = dq_des
            rng = np.random.default_rng()
            print(saved_idx, " / ", nr_trajs)
            start = time.time()
            sol = traj_opt.minimize(
                loss, q0=q0, qT=q1, dq0=dq0, dqT=dqT, mj_model=mj_model, data=data
            )
            # sol = traj_opt.minimize(
            #     loss, q0=q0, dq0=dq0, dqT=dqT, mj_model=mj_model, data=data
            # )
            fail = True if sol.c_best > 1e2 or sol.T_best > 5.0 else False
            if not fail:
                t_comp = time.time() - start
                t_traj = np.arange(0, sol.T_best + 0.1, 0.1)
                q, dq, ddq = sol.get_posvelacc(t_traj)

                x = np.arange(0.1, 1.05, 0.1)
                qe, dqe, ddqe = eval_third_order_poly(x, a_list)
                q = np.concatenate((q, qe.T))
                dq = np.concatenate((dq, dqe.T))
                ddq = np.concatenate((ddq, ddqe.T))

                q = np.concatenate((q, q[-1, :] * np.ones((5, q.shape[1]))))
                dq = np.concatenate((dq, np.zeros((5, dq.shape[1]))))
                ddq = np.concatenate((ddq, np.zeros((5, ddq.shape[1]))))

                q_last = np.copy(q)
                dq_last = np.copy(dq)
                t_best_last = sol.T_best + 1.5

                # with mujoco.viewer.launch_passive(mj_model, data) as viewer:
                #     for pos in q.tolist():
                #         data.qpos[:7] = pos
                #         mujoco.mj_forward(mj_model, data)
                #         viewer.sync()
                #         time.sleep(0.1)

            tail = f"{saved_idx}"
            save_data = {}
            if not fail:
                print()
                print("--------------")
                print(f"Saving data {saved_idx}")
                print("--------------")
                print()
                save_data["q_prev0"] = q_prev0
                save_data["q"] = q
                save_data["dq"] = dq
                save_data["ddq"] = ddq
                np.savez(path + f"traj_vpsto_{tail}.npz", **save_data)
                saved_idx += 1
            else:
                print(f"Failed creating trajectory {saved_idx}")

            # with mujoco.viewer.launch_passive(mj_model, data) as viewer:
            #     while viewer.is_running():
            #         for pos in q:
            #             data.qpos[:7] = pos
            #             mujoco.mj_forward(mj_model, data)
            #             viewer.sync()
            #             time.sleep(0.1)


if __name__ == "__main__":
    main()
