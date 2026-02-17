import time

import casadi as ca
import numpy as np
from acados_template import (
    ACADOS_INFTY,
    AcadosModel,
    AcadosOcp,
    AcadosOcpSolver,
)

from ..ConvexSetFinder import ConvexSetFinder
from ..RobotModel import RobotModel
from ..utils import normalize_set_size


def setup_solver_and_integrator(
    dt: float,
    N: int,
    nr_joints: int,
    smooth: bool,
    use_term: bool,
    creation_mode: str,
) -> AcadosOcpSolver:
    robot_model = RobotModel()
    q = ca.SX.sym("q", nr_joints)
    dq = ca.SX.sym("dq", nr_joints)
    ddq = ca.SX.sym("ddq", nr_joints)
    uprev = ca.SX.sym("uprev", nr_joints)

    max_set_size = 15
    nr_p_col = 10

    x = ca.vertcat(q, dq, ddq, uprev)

    # controls
    u = ca.SX.sym("u", nr_joints)

    # Integration of the system
    q_new = ddq * dt**2 / 2.0 + dq * dt + q + uprev * dt**3 / 8.0 + u * dt**3 / 24.0
    dq_new = ddq * dt + dq + uprev * dt**2 / 3.0 + u * dt**2 / 6.0
    ddq_new = ddq + uprev * dt / 2.0 + u * dt / 2.0
    uprev_new = u

    x_new = ca.vertcat(q_new, dq_new, ddq_new, uprev_new)

    model = AcadosModel()

    model.disc_dyn_expr = x_new
    model.x = x
    model.u = u
    model.name = "flow_safety_model"

    ocp = AcadosOcp()
    ocp.code_export_directory = "/tmp/acados_code"
    ocp.model = model
    ocp.solver_options.N_horizon = N
    ocp.solver_options.tf = N * dt

    ocp.cost.cost_type = "LINEAR_LS"
    ocp.cost.cost_type_e = "LINEAR_LS"

    ocp.model.cost_expr_ext_cost = x
    ocp.model.cost_expr_ext_cost_e = 0

    nx = model.x.rows()
    nu = model.u.rows()
    ny = nx + nu

    ocp.cost.Vx = np.zeros((ny, nx))
    ocp.cost.Vx[:nx, :nx] = np.eye(nx)
    ocp.cost.W = np.eye(ny)
    ocp.cost.W[:nx, :nx] = np.eye(nx)
    if smooth:
        ocp.cost.W[nr_joints : 2 * nr_joints, nr_joints : 2 * nr_joints] *= 0.001
        ocp.cost.W[2 * nr_joints : 3 * nr_joints, 2 * nr_joints : 3 * nr_joints] *= (
            0.001
        )
        ocp.cost.W[3 * nr_joints :, 3 * nr_joints :] *= 0.0001
    else:
        ocp.cost.W[nr_joints:, nr_joints:] *= 0.0
    ocp.cost.W_0 = ocp.cost.W

    Vu = np.zeros((ny, nu))
    Vu[ny - nu :, :] = np.eye(nu)
    ocp.cost.Vu = Vu

    # ocp.cost.Vx_e = np.eye(nx)
    # ocp.cost.W_e = np.eye(nx)
    # if smooth:
    #     ocp.cost.W_e[nr_joints : 2 * nr_joints, nr_joints : 2 * nr_joints] *= 0.001
    #     ocp.cost.W_e[2 * nr_joints : 3 * nr_joints, 2 * nr_joints : 3 * nr_joints] *= (
    #         0.001
    #     )
    # else:
    #     ocp.cost.W_e[nr_joints:, nr_joints:] *= 0.0

    ocp.cost.yref = np.ones((ny,))
    # ocp.cost.yref_e = np.ones((ny_e,))

    ocp.constraints.lbu = robot_model.u_min * np.ones(nu)
    ocp.constraints.ubu = robot_model.u_max * np.ones(nu)
    ocp.constraints.idxbu = np.array(range(nu))

    ocp.constraints.x0 = np.zeros(nx)

    ocp.constraints.lbx = np.concatenate(
        (
            robot_model.q_lim_lower,
            robot_model.dq_lim_lower,
            robot_model.ddq_lim_lower,
            ocp.constraints.lbu,
        )
    )
    ocp.constraints.ubx = np.concatenate(
        (
            robot_model.q_lim_upper,
            robot_model.dq_lim_upper,
            robot_model.ddq_lim_upper,
            ocp.constraints.ubu,
        )
    )
    ocp.constraints.idxbx = np.array(range(nx))

    set_constraints = ca.SX.sym("set_constraint", 0)
    params = ca.SX.sym("params", 0)
    for i in range(nr_p_col):
        p = robot_model.fk_pos_col(q, i)
        a_set = ca.SX.sym("param", max_set_size, 3)
        b_set = ca.SX.sym("param", max_set_size)
        params = ca.vertcat(params, a_set.reshape((-1, 1)), b_set)
        set_constraints = ca.vertcat(set_constraints, a_set @ p - b_set)
    nh = set_constraints.size()[0]
    model.p = params
    # model.p_global = params
    ocp.model.con_h_expr = set_constraints
    ocp.constraints.uh = np.zeros(nh)
    ocp.constraints.lh = -ACADOS_INFTY * np.ones(nh)

    ocp.constraints.idxsh = np.array(range(nh))
    ocp.cost.Zl = np.ones(nh)
    ocp.cost.Zu = np.ones(nh)
    ocp.cost.zl = 1000 * np.ones(nh)
    ocp.cost.zu = 1000 * np.ones(nh)
    ocp.constraints.Jsh = np.eye(nh, dtype=int)

    # Terminal constraint
    if use_term:
        ocp.cost.Vx_e = np.eye(nx)[nr_joints:, :]
        ocp.cost.W_e = 10 * np.eye(nx - nr_joints)
        ocp.cost.yref_e = np.zeros((nx - nr_joints,))

    # set options
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.integrator_type = "DISCRETE"
    ocp.solver_options.print_level = 0
    ocp.solver_options.nlp_solver_type = "SQP_RTI"
    ocp.solver_options.globalization = "FIXED_STEP"
    ocp.solver_options.nlp_solver_max_iter = 50
    ocp.solver_options.nlp_solver_tol_stat = 1e-6
    ocp.solver_options.sim_method_num_steps = 1
    ocp.solver_options.qp_solver_iter_max = 7
    # ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.hessian_approx = "EXACT"
    ocp.parameter_values = np.zeros(params.shape[0])

    # ocp.solver_options.store_iterates = True
    # ocp.solver_options.eval_residual_at_max_iter = True

    ocp_json_file = "/tmp/acados_ocp.json"
    if creation_mode == "cython":
        ocp_solver = AcadosOcpSolver.generate(ocp, json_file=ocp_json_file)
        AcadosOcpSolver.build("/tmp/acados_code", with_cython=True)
        ocp_solver = AcadosOcpSolver.create_cython_solver(ocp_json_file)
    elif creation_mode == "ctypes_precompiled":
        ## Note: skip generate and build assuming this is done before (in cython run)
        ocp_solver = AcadosOcpSolver(
            ocp,
            json_file=ocp_json_file,
            build=False,
            generate=False,
        )
    elif creation_mode == "ctypes_precompiled_no_ocp":
        ocp_solver = AcadosOcpSolver(
            None,
            json_file=ocp_json_file,
            build=False,
            generate=False,
        )
    elif creation_mode == "ctypes":
        AcadosOcpSolver.generate(ocp, json_file=ocp_json_file)
        ocp_solver = AcadosOcpSolver(ocp, json_file=ocp_json_file)
    else:
        raise Exception(f"Invalid creation mode: {creation_mode}")

    # ocp_solver.reset()

    # # integrator
    # sim = AcadosSim()
    #
    # sim.model = model
    # sim.solver_options.integrator_type = "ERK"
    # sim.solver_options.num_stages = 4
    # sim.solver_options.num_steps = 3
    # sim.solver_options.T = 1.0
    # integrator = AcadosSimSolver(sim)

    return ocp_solver


class SafetyFilterAcados:
    def __init__(
        self,
        N=15,
        nr_joints=7,
        dt=0.1,
        smooth=False,
        use_term=False,
        use_sets=False,
        obstacle_manager=None,
        build=True,
        workspace_max=[1.3, 1.3, 1.5],
        workspace_min=[-0.25, -1.3, 0.0],
    ):
        if build:
            creation_mode = "ctypes"
        else:
            creation_mode = "ctypes_precompiled"
        self.ocp_solver = setup_solver_and_integrator(
            dt,
            N,
            nr_joints,
            smooth=smooth,
            use_term=use_term,
            creation_mode=creation_mode,
        )
        self.N = N
        self.nr_joints = nr_joints
        self.use_sets = use_sets
        self.dt = dt
        self.robot_model = RobotModel()
        self.p_human = 100 * np.ones((self.N, 3))
        self.v_human = np.zeros((self.N, 3))

        self.set_finder = ConvexSetFinder(
            obstacle_manager,
            e_max=workspace_max,
            e_min=workspace_min,
            max_set_size=30,
        )

        self.sets_normed = [[]]
        self.reset()

    def reset(self):
        self.t_total = 0
        self.t_loop = 0
        self.t_sets = 0
        self.t_array = []
        self.n_steps = 0
        self.q = np.zeros(self.nr_joints)
        self.dq = np.zeros(self.nr_joints)
        self.ddq = np.zeros(self.nr_joints)
        self.dddq = np.zeros(self.nr_joints)
        self.q_last = np.zeros((self.nr_joints, self.N + 1))
        self.dq_last = np.zeros((self.nr_joints, self.N + 1))
        self.ddq_last = np.zeros((self.nr_joints, self.N + 1))
        self.dddq_last = np.zeros((self.nr_joints, self.N + 1))
        self.updated = True

    def compute_sets(self, q_set0, q_setf):
        nr_p_col = len(self.robot_model.col_ids)
        p_list = [self.robot_model.fk_pos_col(q_set0, i) for i in range(nr_p_col)]
        p_list_f = [self.robot_model.fk_pos_col(q_setf, i) for i in range(nr_p_col)]
        set_joints = []
        joint_sizes = self.robot_model.col_joint_sizes
        time_start_sets = time.perf_counter()
        set_joints = [
            [
                *self.set_finder.find_set_collision_avoidance(
                    pl, pf, limit_space=False, e_max=0.7
                )[:2]
            ]
            for pl, pf in zip(p_list, p_list_f)
        ]
        set_joints = [[a, b - s] for (a, b), s in zip(set_joints, joint_sizes)]
        # for i, (pl, pf) in enumerate(zip(p_list, p_list_f)):
        #     a_c, b_c, _ = self.set_finder.find_set_collision_avoidance(
        #         pl, pf, limit_space=False, e_max=0.7
        #     )
        #     set_joints.append([a_c, b_c - joint_sizes[i]])

        self.sets_normed = normalize_set_size(set_joints, 15)
        self.a_set_joints = [x[0] for x in self.sets_normed]
        self.b_set_joints = np.array([x[1] for x in self.sets_normed])
        time_sets = time.perf_counter() - time_start_sets
        self.t_sets += time_sets
        params = np.concatenate(
            [
                np.concatenate([a.T.flatten(), b])
                for a, b in zip(self.a_set_joints, self.b_set_joints)
            ]
        )
        return params

    def step(self, q0, q_des, dq0=None, ddq0=None, dddq0=None, update=True):
        time_loop_start = time.perf_counter()
        if self.n_steps == 0:
            self.q_last = np.vstack([q0] * (self.N + 1)).T
        if dq0 is None:
            dq0 = self.dq
            ddq0 = self.ddq
            dddq0 = self.dddq

        if self.use_sets and self.updated:
            time_start_sets = time.perf_counter()
            q_set0 = q0
            q_setf = self.q_last[:, -2]
            self.params_sets = self.compute_sets(q_set0, q_setf)
            # self.params_sets = []
            # for i in range(self.N):
            #     if self.n_steps == 0:
            #         q_set0 = q0
            #         q_setf = q0
            #     else:
            #         q_set0 = self.q_last[:, i]
            #         q_setf = self.q_last[:, i + 1]
            #     self.params_sets.append(self.compute_sets(q_set0, q_setf))
            time_sets = time.perf_counter() - time_start_sets
            self.t_sets = time_sets

        x0 = np.concatenate((q0, dq0, ddq0, dddq0))
        self.ocp_solver.set(0, "x", x0)
        self.ocp_solver.set(0, "lbx", x0)
        self.ocp_solver.set(0, "ubx", x0)

        if not np.any(self.q_last):
            for i in range(1, self.N + 1):
                self.ocp_solver.set(i, "x", x0)
        else:
            x_mat = np.vstack(
                (self.q_last, self.dq_last, self.ddq_last, self.dddq_last)
            )
            for i in range(1, self.N + 1):
                self.ocp_solver.set(i, "x", x_mat[:, i])

        # Set reference
        yref_zero_tail = np.zeros((self.N, 4 * self.nr_joints))
        yref = np.hstack((q_des, yref_zero_tail))
        for i in range(1, self.N):
            self.ocp_solver.set(i, "yref", yref[i, :])

        # Set parameters
        if self.use_sets:
            for i in range(1, self.N):
                # self.ocp_solver.set(i, "p", self.params_sets[i])
                self.ocp_solver.set(i, "p", self.params_sets)

        time_start = time.perf_counter()
        status = self.ocp_solver.solve()
        time_elapsed = time.perf_counter() - time_start
        self.t_total += time_elapsed
        self.t_array.append(time_elapsed)
        self.n_steps += 1
        # self.ocp_solver.print_statistics()  # encapsulates: stat = ocp_solver.get_stats("statistics")
        # qp_iter = self.ocp_solver.get_stats("qp_iter")
        # print(f"acados returned status {status} with {sqp_iter} sqp iterations.")
        max_slack = np.max(
            np.array([self.ocp_solver.get(k, "su") for k in range(1, self.N)])
        )
        if update:
            self.updated = False
        if max_slack > 1e-3 or status > 0:
            raise RuntimeError(f"SQP Solve did not succeed, max slack {max_slack}")
        else:
            x_opt = np.array([self.ocp_solver.get(i, "x") for i in range(self.N + 1)])
            u_opt = np.array([self.ocp_solver.get(i, "u") for i in range(self.N)]).T
            q_opt = x_opt[:, : self.nr_joints].T
            dq_opt = x_opt[:, self.nr_joints : 2 * self.nr_joints].T
            ddq_opt = x_opt[:, 2 * self.nr_joints : 3 * self.nr_joints].T

        if update:
            self.q_last = q_opt
            self.dq_last = dq_opt
            self.ddq_last = ddq_opt
            self.dddq_last = np.hstack((dddq0[:, None], u_opt))

        self.t_loop += time.perf_counter() - time_loop_start
        return q_opt[:, :-1]

    def update_from_last_solution(self, n_actions):
        self.q_last = np.hstack((self.q_last[:, 1:], self.q_last[:, -1:]))
        self.dq_last = np.hstack((self.dq_last[:, 1:], self.dq_last[:, -1:]))
        self.ddq_last = np.hstack((self.ddq_last[:, 1:], self.ddq_last[:, -1:]))
        self.dddq_last = np.hstack((self.dddq_last[:, 1:], self.dddq_last[:, -1:]))

    def update_initial_state(self, n_actions):
        self.updated = True
        self.t_array = []
        self.q = self.q_last[:, n_actions]
        self.dq = self.dq_last[:, n_actions]
        self.ddq = self.ddq_last[:, n_actions]
        self.dddq = self.dddq_last[:, n_actions]
