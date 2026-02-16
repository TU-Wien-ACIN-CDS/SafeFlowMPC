import os
from pathlib import Path

import casadi as ca
import numpy as np
import pinocchio as pin
from scipy.spatial.transform import Rotation as R

GEN_CA = False
current_dir = os.path.dirname(os.path.abspath(__file__))
# Define the absolute path for saving casadi functions
CA_SAVE_PATH = os.path.join(current_dir, "")
USE_IIWA = True
if GEN_CA:
    from pinocchio import casadi as cpin


class RobotModel:
    def __init__(self):
        model_path = Path(__file__).parent
        if USE_IIWA:
            urdf_filename = "iiwa.urdf"
        else:
            urdf_filename = "gen3_arm.urdf"
        urdf_model_path = model_path / urdf_filename
        self.model, collision_model, visual_model = pin.buildModelsFromUrdf(
            urdf_model_path, package_dirs=model_path
        )
        self.ee_id = self.model.getFrameId("end_effector_link")
        self.col_names = [
            "joint_3",
            "joint_4",
            "joint_5",
            "joint_6",
            "link7_col_link",
            "link4_col_link",
            # "end_effector_col_link_0",
            "end_effector_col_link_b1",
            "end_effector_col_link_b2",
            "end_effector_col_link_1",
            "end_effector_col_link_2",
        ]
        self.col_ids = []
        for i in range(len(self.col_names)):
            if i < 4:
                self.col_ids.append(self.model.getJointId(self.col_names[i]))
            else:
                self.col_ids.append(self.model.getFrameId(self.col_names[i]))
        if USE_IIWA:
            self.col_joint_sizes = [
                0.09,
                0.1,
                0.07,
                0.08,
                0.06,
                0.08,
                0.03,
                0.03,
                0.03,
                0.03,
            ]
        else:
            self.col_joint_sizes = [0.09, 0.09, 0.06, 0.06, 0.06, 0.06, 0.075]
        if GEN_CA:
            self.cmodel = cpin.Model(self.model)
        # [self.cmodel.names[i] for i in range(7)]
        # Joint limits
        self.q_lim_lower = self.model.lowerPositionLimit + 3 * np.pi / 180
        self.q_lim_upper = self.model.upperPositionLimit - 3 * np.pi / 180
        if not USE_IIWA:
            self.q_lim_lower[[0, 2, 4, 6]] = -np.inf
            self.q_lim_upper[[0, 2, 4, 6]] = np.inf
        self.dq_lim_lower = -self.model.velocityLimit * 0.92
        self.dq_lim_upper = self.model.velocityLimit * 0.92
        self.tau_lim_lower = [-320, -320, -176, -176, -110, -40, -40]
        self.tau_lim_upper = [320, 320, 176, 176, 110, 40, 40]
        self.ddq_lim_lower = -5 * np.ones(7)
        self.ddq_lim_upper = 5 * np.ones(7)
        self.u_max = 15
        self.u_min = -15

        self.setup_ik_problem()

    def get_robot_limits(self):
        return (
            self.q_lim_upper,
            self.q_lim_lower,
            self.dq_lim_upper,
            self.dq_lim_lower,
            self.tau_lim_upper,
            self.tau_lim_lower,
            self.u_max,
            self.u_min,
        )

    def forward_kinematics(self, q: np.ndarray, dq: np.ndarray):
        """Compute forward kinematics to get the position of the end effector in
        cartesian space. Also computes the jacobian and its derivative.
        """
        jac_ee = self.jacobian_fk(q)
        djac_ee = self.djacobian_fk(q, dq)
        p_robot = self.fk(q)
        return p_robot, jac_ee, djac_ee

    def setup_ik_problem(self):
        q = ca.SX.sym("q", 7)
        pd = ca.SX.sym("p desired", 3)
        rd = ca.SX.sym("r desired", 3, 3)
        r_ee = self.hom_transform_endeffector(q)[:3, :3]
        J = ca.sumsqr(self.fk_pos(q) - pd[:3])
        J += ca.sumsqr(r_ee @ rd.T - ca.SX.eye(3))
        w = [q]
        lbw = self.q_lim_lower
        ubw = self.q_lim_upper

        params = ca.vertcat(pd, rd.reshape((-1, 1)))

        prob = {
            "f": J,
            "x": ca.vertcat(*w),
            # 'g': ca.vertcat(*g),
            "p": params,
        }
        self.lbu = lbw
        self.ubu = ubw
        # lbg = lbg
        # ubg = ubg
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
        self.ik_solver = ca.nlpsol("solver", "ipopt", prob, solver_opts)

    def inverse_kinematics(self, pd, rd, q0):
        """Inverse kinematics based on optimization."""
        params = np.concatenate((pd, rd.T.flatten()))
        sol = self.ik_solver(x0=q0, lbx=self.lbu, ubx=self.ubu, p=params)
        q_ik = np.array(sol["x"]).flatten()
        if not self.ik_solver.stats()["success"]:
            print("(IK) ERROR No convergence in IK optimization")
        h_ik = self.hom_transform_endeffector(q_ik)
        pos_error = np.linalg.norm(pd - h_ik[:3, 3])
        rot_error = np.linalg.norm(R.from_matrix(h_ik[:3, :3] @ rd.T).as_rotvec())
        # print(f"(IK) Position error {pos_error}m")
        # print(f"(IK) Rotation error {rot_error * 180 / np.pi} deg")
        return q_ik, pos_error, rot_error

    def fk_pos(self, q):
        if isinstance(q, np.ndarray):
            data = self.model.createData()
            pin.framesForwardKinematics(self.model, data, q)
        elif GEN_CA:
            data = self.cmodel.createData()
            cpin.framesForwardKinematics(self.cmodel, data, q)
        if isinstance(q, np.ndarray) or GEN_CA:
            p = data.oMf[self.ee_id].translation
            if GEN_CA and not isinstance(q, np.ndarray):
                ca.Function("p", [q], [p]).save(f"{CA_SAVE_PATH}fk_pos.ca")
        else:
            p_fun = ca.Function.load(f"{CA_SAVE_PATH}fk_pos.ca")
            p = p_fun(q)
        return p

    def fk_pos_col(self, q, i):
        if isinstance(q, np.ndarray):
            data = self.model.createData()
            pin.forwardKinematics(self.model, data, q)
            pin.framesForwardKinematics(self.model, data, q)
        elif GEN_CA:
            data = self.cmodel.createData()
            cpin.forwardKinematics(self.cmodel, data, q)
            cpin.framesForwardKinematics(self.cmodel, data, q)
        if isinstance(q, np.ndarray) or GEN_CA:
            if i < 4:
                p = data.oMi[self.col_ids[i]].translation
            else:
                p = data.oMf[self.col_ids[i]].translation
            if GEN_CA and not isinstance(q, np.ndarray):
                ca.Function("p", [q], [p]).save(f"{CA_SAVE_PATH}fk_pos_col_{i}.ca")
        else:
            p_fun = ca.Function.load(f"{CA_SAVE_PATH}fk_pos_col_{i}.ca")
            p = p_fun(q)
        return p

    def fk(self, q):
        """Compute the end effector position of the robot in cartesian space
        given the joint configuration.
        """
        if isinstance(q, np.ndarray):
            m = np.zeros(6)
        else:
            m = ca.SX.zeros(6)
        h = self.hom_transform_endeffector(q)
        m[:3] = h[:3, 3]
        m[3:] = R.from_matrix(h[:3, :3]).as_rotvec()

        return m

    def hom_transform_endeffector(self, q):
        if isinstance(q, np.ndarray):
            data = self.model.createData()
            pin.framesForwardKinematics(self.model, data, q)
        elif GEN_CA:
            data = self.cmodel.createData()
            cpin.framesForwardKinematics(self.cmodel, data, q)
        if isinstance(q, np.ndarray) or GEN_CA:
            p = data.oMf[self.ee_id].homogeneous
            if GEN_CA and not isinstance(q, np.ndarray):
                ca.Function("p", [q], [p]).save(f"{CA_SAVE_PATH}hom_trans.ca")
        else:
            p_fun = ca.Function.load(f"{CA_SAVE_PATH}hom_trans.ca")
            p = p_fun(q)
        return p

    def jacobian_fk(self, q):
        if isinstance(q, np.ndarray):
            data = self.model.createData()
            pin.computeForwardKinematicsDerivatives(self.model, data, q, q, q)
            jac = pin.getFrameJacobian(
                self.model, data, self.ee_id, pin.LOCAL_WORLD_ALIGNED
            )
        else:
            if GEN_CA:
                data = self.cmodel.createData()
                cpin.computeForwardKinematicsDerivatives(self.cmodel, data, q, q, q)
                jac = cpin.getFrameJacobian(
                    self.cmodel, data, self.ee_id, pin.LOCAL_WORLD_ALIGNED
                )
                ca.Function("p", [q], [jac]).save(f"{CA_SAVE_PATH}jacobian.ca")
            else:
                jac_fun = ca.Function.load(f"{CA_SAVE_PATH}jacobian.ca")
                jac = jac_fun(q)
        return jac

    def djacobian_fk(self, q, dq):
        if isinstance(q, np.ndarray):
            data = self.model.createData()
            pin.computeForwardKinematicsDerivatives(self.model, data, q, dq, dq)
            djac = pin.getFrameJacobianTimeVariation(
                self.model, data, self.ee_id, pin.LOCAL_WORLD_ALIGNED
            )
        else:
            data = self.cmodel.createData()
            cpin.computeForwardKinematicsDerivatives(self.cmodel, data, q, dq, dq)
            if GEN_CA:
                djac = cpin.getFrameJacobianTimeVariation(
                    self.cmodel, data, self.ee_id, pin.LOCAL_WORLD_ALIGNED
                )
                ca.Function("p", [q, dq], [djac]).save(f"{CA_SAVE_PATH}djacobian.ca")
            else:
                djac_fun = ca.Function.load(f"{CA_SAVE_PATH}djacobian.ca")
                djac = djac_fun(q, dq)
        return djac

    def velocity_ee(self, q, dq):
        jac = self.jacobian_fk(q)
        v = jac @ dq
        return v[:3]

    def acceleration_ee(self, q, dq, ddq):
        jac = self.jacobian_fk(q)
        djac = self.djacobian_fk(q, dq)
        a = djac @ dq + jac @ ddq
        return a

    def omega_ee(self, q, dq):
        jac = self.jacobian_fk(q)
        w = jac @ dq
        return w[3:]


if __name__ == "__main__":
    model = RobotModel()
    p = model.fk(np.zeros(7))
    model.fk(np.zeros(7))
    model.velocity_ee(np.zeros(7), np.zeros(7))
