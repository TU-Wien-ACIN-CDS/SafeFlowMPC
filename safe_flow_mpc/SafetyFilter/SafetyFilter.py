import copy
import time

import casadi as ca
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R

from ..ConvexSetFinder import ConvexSetFinder
from ..RobotModel import RobotModel
from ..utils import normalize_set_size


def qp_safe_problem(N, nr_joints, dt, smooth=False, use_term=False, use_sets=False):
    """Build the optimization problem using symbolic variables such that it
    can be easily used later on by just inputting the new state.
    """

    q_des = ca.SX.sym("x_des", N, nr_joints)

    # Initialize variables
    J = 0
    g = []
    lbg = []
    ubg = []

    # Sets for collision avoidance
    max_set_size = 15
    nr_p_col = 7
    p0s = ca.SX.sym("p0s", 3, nr_p_col, N)
    q0s = ca.SX.sym("q0s", N, nr_joints)
    jacobians = ca.SX.sym("q0s", 3, nr_joints, N)
    a_set_joints = ca.SX.sym("a_set_joints", max_set_size, 3, nr_p_col)
    b_set_joints = ca.SX.sym("b_set_joints", nr_p_col, max_set_size)

    # states
    q = ca.SX.sym("q", N, nr_joints)
    dq = ca.SX.sym("dq", N, nr_joints)
    ddq = ca.SX.sym("ddq", N, nr_joints)
    u = ca.SX.sym("u", N, nr_joints)

    w = ca.vertcat(
        q[:],
        dq[:],
        ddq[:],
        u[:],
    )
    if use_sets:
        slacks = ca.SX.sym("s sets", nr_p_col)
        w = ca.vertcat(w, slacks)

    for k in range(N - 1):
        q_new = (
            ddq[k, :] * dt**2 / 2.0
            + dq[k, :] * dt
            + q[k, :]
            + u[k, :7] * dt**3 / 8.0
            + u[k + 1, :7] * dt**3 / 24.0
        )
        dq_new = (
            ddq[k, :] * dt
            + dq[k, :]
            + u[k, :7] * dt**2 / 3.0
            + u[k + 1, :7] * dt**2 / 6.0
        )
        ddq_new = ddq[k, :] + u[k, :7] * dt / 2.0 + u[k + 1, :7] * dt / 2.0

        # Dynamical system constraint
        g += [q_new - q[k + 1, :]]
        lbg += [0] * nr_joints
        ubg += [0] * nr_joints
        g += [dq_new - dq[k + 1, :]]
        lbg += [0] * nr_joints
        ubg += [0] * nr_joints
        g += [ddq_new - ddq[k + 1, :]]
        lbg += [0] * nr_joints
        ubg += [0] * nr_joints

    # Formulate the NLP
    for k in range(1, N):
        # Increment objective value
        if not use_term:
            J = J + ca.sumsqr(q[k, :].T - q_des[k, :].T)
            if smooth:
                J = J + 0.001 * ca.sumsqr(dq[k, :])
                J = J + 0.001 * ca.sumsqr(ddq[k, :])
                J = J + 0.0001 * ca.sumsqr(u[k, :])
        else:
            if k <= 3:
                J = J + ca.sumsqr(q[k, :].T - q_des[k, :].T)
                if smooth:
                    J = J + 0.001 * ca.sumsqr(dq[k, :])
                    J = J + 0.001 * ca.sumsqr(ddq[k, :])
                    J = J + 0.0001 * ca.sumsqr(u[k, :])
            else:
                J = J + 1 / k * ca.sumsqr(q[k, :].T - q_des[k, :].T)
                if smooth:
                    J = J + 1 / k * 0.001 * ca.sumsqr(dq[k, :])
                    J = J + 1 / k * 0.001 * ca.sumsqr(ddq[k, :])
                    J = J + 1 / k * 0.0001 * ca.sumsqr(u[k, :])

            # if k == 1:
            #     g += [q_des[k, :] - q[k, :]]
            #     lbg += [0] * nr_joints
            #     ubg += [0] * nr_joints

        # -----------------------------------------------------------------
        # CONSTRAINTS
        # -----------------------------------------------------------------
        if use_sets and k == 2:
            # for i in range(nr_p_col):
            for i in range(nr_p_col):
                aj = a_set_joints[i]
                bj = b_set_joints[i, :]
                g += [
                    (aj @ (p0s[k][:, i] + jacobians[k] @ (q[k, :] - q0s[k, :]).T)).T
                    - bj
                    - slacks[i]
                ]
                lbg += [-np.inf] * max_set_size
                ubg += [0] * max_set_size
            J = J + 1 * ca.sumsqr(slacks)

        # Terminal constraints
        if use_term and k == N - 1:
            J += 100 * ca.sumsqr(dq[k, :])
            J += 100 * ca.sumsqr(ddq[k, :])
            J += 100 * ca.sumsqr(u[k, :])
            # g += [dq[k, :]]
            # g += [ddq[k, :]]
            # g += [u[k, :]]
            # lbg += [0] * 3 * nr_joints
            # ubg += [0] * 3 * nr_joints

    # Create a QP solver
    if not use_sets:
        params = ca.vertcat(q_des.reshape((-1, 1)))
    else:
        params = ca.vertcat(q_des.reshape((-1, 1)), q0s.reshape((-1, 1)))
        for i in range(len(p0s)):
            params = ca.vertcat(params, p0s[i].reshape((-1, 1)))
        for i in range(len(jacobians)):
            params = ca.vertcat(params, jacobians[i].reshape((-1, 1)))
        for i in range(len(a_set_joints)):
            params = ca.vertcat(params, a_set_joints[i].reshape((-1, 1)))
        params = ca.vertcat(params, b_set_joints.reshape((-1, 1)))

    prob = {"f": J, "x": w, "g": ca.horzcat(*g), "p": params}

    # solver = ca.qpsol(
    #     "solver", "qpoases", prob, {"print_time": False, "printLevel": "none"}
    # )
    # solver = ca.qpsol(
    #     "solver",
    #     "osqp",
    #     prob,
    #     {
    #         "print_time": False,
    #         "osqp": {
    #             "verbose": False,
    #             "eps_abs": 1e-6,
    #             "eps_rel": 1e-6,
    #         },
    #     },
    # )
    solver = ca.qpsol(
        "solver",
        "ipqp",
        prob,
        {
            "print_time": False,
            "print_header": False,
            "print_iter": False,
            "linear_solver": "qr",
        },
    )

    return solver, lbg, ubg


class SafetyFilter:
    def __init__(
        self,
        N=15,
        nr_joints=7,
        dt=0.1,
        smooth=False,
        use_term=False,
        use_sets=False,
        obstacle_manager=None,
    ):
        self.solver, self.lbg, self.ubg = qp_safe_problem(
            N, nr_joints, dt, smooth, use_term, use_sets=use_sets
        )
        self.N = N
        self.nr_joints = nr_joints
        self.use_sets = use_sets
        self.dt = dt
        self.robot_model = RobotModel()

        self.set_finder = ConvexSetFinder(
            obstacle_manager,
            e_max=[1.3, 1.3, 1.5],
            e_min=[-0.25, -1.3, 0.0],
            max_set_size=30,
        )
        self.nr_slacks = 7
        self.reset()

    def reset(self):
        self.t_total = 0
        self.n_steps = 0
        self.q = np.zeros(self.nr_joints)
        self.qf = np.zeros(self.nr_joints)
        self.dq = np.zeros(self.nr_joints)
        self.ddq = np.zeros(self.nr_joints)
        self.dddq = np.zeros(self.nr_joints)
        self.q_last = np.zeros((self.nr_joints, self.N))
        self.dq_last = np.zeros((self.nr_joints, self.N))
        self.ddq_last = np.zeros((self.nr_joints, self.N))
        self.dddq_last = np.zeros((self.nr_joints, self.N))
        self.updated = True

    def compute_state(self, u, q0, dq0, ddq0):
        q = np.empty((self.N, self.nr_joints))
        dq = np.empty((self.N, self.nr_joints))
        ddq = np.empty((self.N, self.nr_joints))
        q[0, :] = q0
        dq[0, :] = dq0
        ddq[0, :] = ddq0

        for k in range(self.N - 1):
            q[k + 1, :] = (
                ddq[k, :] * self.dt**2 / 2.0
                + dq[k, :] * self.dt
                + q[k, :]
                + u[k, :] * self.dt**3 / 8.0
                + u[k + 1, :] * self.dt**3 / 24.0
            )
            dq[k + 1, :] = (
                ddq[k, :] * self.dt
                + dq[k, :]
                + u[k, :] * self.dt**2 / 3.0
                + u[k + 1, :] * self.dt**2 / 6.0
            )
            ddq[k + 1, :] = (
                ddq[k, :] + u[k, :] * self.dt / 2.0 + u[k + 1, :] * self.dt / 2.0
            )

        return q, dq, ddq

    def step(self, q0, q_des, dq0=None, ddq0=None, dddq0=None, plot=False, update=True):
        if self.n_steps == 0:
            self.qf = q0
        if dq0 is None:
            dq0 = self.dq
            ddq0 = self.ddq
            dddq0 = self.dddq

        # Limit q_des to joint limits
        for i in range(self.N):
            q_des[i, :] = np.maximum(q_des[i, :], self.robot_model.q_lim_lower)
            q_des[i, :] = np.minimum(q_des[i, :], self.robot_model.q_lim_upper)

        x = np.copy(
            np.concatenate(
                (
                    self.q_last.T.flatten(),
                    self.dq_last.T.flatten(),
                    self.ddq_last.T.flatten(),
                    self.dddq_last.T.flatten(),
                )
            )
        )

        if self.use_sets:
            if self.updated:
                p_list = [self.robot_model.fk_pos_col(q0, i) for i in range(7)]
                p_list_f = [self.robot_model.fk_pos_col(self.qf, i) for i in range(7)]
                # p_list = [self.robot_model.fk_pos_col(self.q, i) for i in [3]]
                # p_list_f = [self.robot_model.fk_pos_col(self.qf, i) for i in [3]]
                set_joints = []
                joint_sizes = self.robot_model.col_joint_sizes
                for i, (pl, pf) in enumerate(zip(p_list, p_list_f)):
                    a_c, b_c, _ = self.set_finder.find_set_collision_avoidance(
                        pl, pf, limit_space=True, e_max=0.7
                    )
                    set_joints.append([a_c, b_c - joint_sizes[i]])

                sets_normed = normalize_set_size(set_joints, 15)
                self.a_set_joints = [x[0] for x in sets_normed]
                self.b_set_joints = np.array([x[1] for x in sets_normed])

            if self.n_steps == 0:
                self.q0s = np.vstack([q0] * 15)
            else:
                self.q0s = self.q_last.T
            self.p0s = []
            for i in range(self.q0s.shape[0]):
                self.p0s.append(
                    np.array([self.robot_model.fk_pos_col(q0, i) for i in range(7)]).T
                )

            self.jacobians = []
            for i in range(self.q0s.shape[0]):
                self.jacobians.append(
                    self.robot_model.jacobian_fk(self.q0s[i, :])[:3, :]
                )

        q_ub = self.robot_model.q_lim_upper
        q_lb = self.robot_model.q_lim_lower
        q_ub = np.repeat(q_ub, self.N)
        q_lb = np.repeat(q_lb, self.N)

        dq_ub = self.robot_model.dq_lim_upper
        dq_lb = self.robot_model.dq_lim_lower
        dq_ub = np.repeat(dq_ub, self.N)
        dq_lb = np.repeat(dq_lb, self.N)

        ddq_ub = 5.0 * np.ones(self.N * self.nr_joints)
        ddq_lb = -ddq_ub

        u_ub = self.robot_model.u_max * np.ones(self.N * self.nr_joints)
        u_lb = self.robot_model.u_min * np.ones(self.N * self.nr_joints)

        q_lb[0 : -1 : self.N] = q0
        dq_lb[0 : -1 : self.N] = dq0
        ddq_lb[0 : -1 : self.N] = ddq0
        u_lb[0 : -1 : self.N] = dddq0

        lbx = np.concatenate([q_lb, dq_lb, ddq_lb, u_lb])
        if self.use_sets:
            lbx = np.concatenate([lbx, np.zeros(self.nr_slacks)])

        q_ub[0 : -1 : self.N] = q0
        dq_ub[0 : -1 : self.N] = dq0
        ddq_ub[0 : -1 : self.N] = ddq0
        u_ub[0 : -1 : self.N] = dddq0

        ubx = np.concatenate([q_ub, dq_ub, ddq_ub, u_ub])
        if self.use_sets:
            ubx = np.concatenate([ubx, np.inf * np.ones(self.nr_slacks)])

        if not self.use_sets:
            params = q_des.T.flatten()
        else:
            params = np.concatenate((q_des.T.flatten(), self.q0s.T.flatten()))
            for i in range(len(self.p0s)):
                params = np.concatenate((params, self.p0s[i].T.flatten()))
            for i in range(len(self.jacobians)):
                params = np.concatenate((params, self.jacobians[i].T.flatten()))
            for i in range(len(self.a_set_joints)):
                params = np.concatenate((params, self.a_set_joints[i].T.flatten()))
            params = np.concatenate((params, self.b_set_joints.T.flatten()))

        if self.use_sets:
            x = np.concatenate((x, np.zeros(self.nr_slacks)))

        time_start = time.perf_counter()
        sol = self.solver(x0=x, lbx=lbx, ubx=ubx, lbg=self.lbg, ubg=self.ubg, p=params)
        time_elapsed = time.perf_counter() - time_start
        self.t_total += time_elapsed
        self.n_steps += 1
        # stats = self.solver.stats()
        # print()
        # print("Safety QP")
        # print("------")
        # print("Time")
        # print(time_elapsed)
        # print("Success")
        # print(stats["success"])
        # print("Iters")
        # print(stats["iter_count"])
        x_opt = sol["x"].full().flatten()
        slacks = x_opt[-self.nr_slacks :]
        q_opt = np.reshape(x_opt[0 : 7 * self.N], (self.N, 7), "F").T
        dq_opt = np.reshape(x_opt[7 * self.N : 2 * 7 * self.N], (self.N, 7), "F").T
        ddq_opt = np.reshape(x_opt[2 * 7 * self.N : 3 * 7 * self.N], (self.N, 7), "F").T
        u_opt = np.reshape(x_opt[3 * 7 * self.N : 4 * 7 * self.N], (self.N, 7), "F").T
        if update:
            self.q_last = q_opt
            self.dq_last = dq_opt
            self.ddq_last = ddq_opt
            self.dddq_last = u_opt
        if update:
            self.updated = False
        # p_opt = np.empty((6, q_opt.shape[1]))
        # for i in range(q_opt.shape[1]):
        #     p_opt[:, i], _, _ = robot_model.forward_kinematics(
        #         q_opt[:, i], 0 * q_opt[:, i]
        #     )

        # aj = self.a_set_joints[1]
        # bj = self.b_set_joints[1]
        # dj = (
        #     (
        #         aj
        #         @ (
        #             self.p0s[5][:, 1]
        #             + self.jacobians[5] @ (q_opt[:, 5] - self.q0s[5, :])
        #         )
        #     ).T
        #     - bj
        #     - slacks[1]
        # )
        # dj + slacks[1]

        # g = np.array(sol["g"]).flatten()
        # g_viol = -np.sum(g[np.where(g < np.array(self.lbg) - 1e-6)[0]])
        # g_viol += np.sum(g[np.where(g > np.array(self.ubg) + 1e-6)[0]])
        # print(g_viol)

        if plot:
            plt.figure()
            for i in range(7):
                plt.subplot(2, 4, i + 1)
                plt.plot(self.q_last[i, :], f"C{i}")
                plt.plot(q_des[:, i], f"C{i}--")
            plt.show()

        return q_opt

    def update_initial_state(self, n_actions):
        self.updated = True
        self.q = self.q_last[:, n_actions]
        self.qf = self.q_last[:, -1]
        self.dq = self.dq_last[:, n_actions]
        self.ddq = self.ddq_last[:, n_actions]
        self.dddq = self.dddq_last[:, n_actions]


if __name__ == "__main__":
    N = 15
    nr_joints = 7
    dt = 0.1
    robot_model = RobotModel()

    q0 = np.zeros(nr_joints)
    dq0 = np.zeros(nr_joints)
    ddq0 = np.zeros(nr_joints)
    dddq0 = np.zeros(nr_joints)
    q0[1] = 0.0
    q0[3] = -np.pi / 2
    q0[5] = np.pi / 2
    p0, _, _ = robot_model.forward_kinematics(q0, 0 * q0)
    q_des = np.empty((N, 7))
    q_des[0, :] = q0
    p1 = np.copy(p0[:3])
    r1 = R.from_rotvec(p0[3:]).as_matrix()
    for i in range(1, q_des.shape[0]):
        p1[0] += 0.01
        p1[2] += 0.01
        qdc, _, _ = robot_model.inverse_kinematics(p1, r1, q_des[i - 1, :])
        q_des[i, :] = qdc
    u = np.zeros((N, nr_joints))

    sf = SafetyFilter()
    p_opt, q_opt, dq_opt, ddq_opt, u_opt = sf.step(q0, q_des, dq0, ddq0, dddq0)

    plt.figure()
    for i in range(7):
        plt.subplot(4, 7, i + 1)
        plt.plot(q_opt[i, :])
        plt.plot(q_des.T[i, :], "--")
        plt.subplot(4, 7, i + 8)
        plt.plot(dq_opt[i, :])
        plt.subplot(4, 7, i + 15)
        plt.plot(ddq_opt[i, :])
        plt.subplot(4, 7, i + 22)
        plt.plot(u_opt[i, :])

    # plt.figure()
    # for i in range(6):
    #     plt.subplot(2, 3, i + 1)
    #     plt.plot(p_opt[i, :])
    plt.show()
