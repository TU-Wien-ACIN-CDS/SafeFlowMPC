import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
from scipy.optimize import linprog
from torch.profiler import ProfilerActivity, profile, record_function

from ..utils.util_functions import plot_set


@triton.jit
def iter_separation_kernel(
    pointcloud_ptr,
    targets_ptr,
    mask_ptr,
    normals_ptr,
    bs_ptr,
    N,
    M,
    BLOCK_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Load targets and points
    targets = tl.load(targets_ptr + offs_m[:, None] * 3 + 0, 2)
    pts = tl.load(pointcloud_ptr + offs_n[:, None] * 3 + 0, 2)

    # Mask & distance calc
    mask = tl.load(mask_ptr + offs_m[:, None] * N + offs_n)
    diff = targets[:, None] - pts
    dist2 = tl.sum(diff * diff, axis=2)
    dist2 = tl.where(mask, dist2, 1e9 + tl.zeros_like(dist2))

    # Argmin across N dimension
    # min_dist2 = tl.minimum(dist2, axis=1)
    min_idx = tl.argmin(dist2, axis=1)

    # Compute plane normal + b
    closest_pts = tl.load(pointcloud_ptr + min_idx[:, None] * 3 + 0, 2)
    norm = closest_pts - targets
    norm = norm / tl.sqrt(tl.sum(norm * norm, axis=1) + 1e-6)
    b_val = tl.sum(norm * closest_pts, axis=1)

    # Write normals and bs
    tl.store(normals_ptr + pid_n * BLOCK_N * 3 + offs_n[:, None] * 3, norm)
    tl.store(bs_ptr + pid_m * BLOCK_M + offs_m, b_val)

    # Update mask: dot = pts·norm^T - b
    dot = tl.sum(pts * norm[:, None, :], axis=2) - b_val[:, None]
    new_mask = tl.where(dot < -1e-6, mask, False)
    tl.store(mask_ptr + offs_m[:, None] * N + offs_n, new_mask)


def sample_boundary_points(A, b, num_samples=100):
    n = A.shape[1]
    samples = []

    for i in range(len(b)):
        Ai = A[i]
        bi = b[i]

        # Define the facet: A_i x = b_i, A x <= b
        # We'll sample in the null space of Ai (tangent to the facet)
        null_space = nullspace(Ai.reshape(1, -1))  # Basis for tangent space

        if null_space.size == 0:
            continue  # Degenerate facet

        # Find a starting point on the facet by solving LP: A x <= b, A_i x = b_i
        x0 = find_point_on_facet(A, b, Ai, bi)
        if x0 is None:
            continue

        # Hit-and-Run on the facet
        for _ in range(num_samples // len(b)):
            direction = null_space @ np.random.randn(null_space.shape[1])
            direction = direction / np.linalg.norm(direction)

            t_min, t_max = find_step_limits(x0, direction, A, b, Ai, bi)

            if t_max > t_min:
                t = np.random.uniform(t_min, t_max)
                x_new = x0 + t * direction
                x0 = x_new
                samples.append(x_new)

    return np.array(samples)


def nullspace(A, rtol=1e-5):
    # Null space of A: all x such that A @ x = 0
    u, s, vh = np.linalg.svd(A)
    rank = (s > rtol * s[0]).sum()
    return vh[rank:].T


def find_point_on_facet(A, b, Ai, bi):
    n = A.shape[1]
    c = np.zeros(n)  # Any objective, just need feasibility

    bounds = [(None, None)] * n
    constraints = [
        {"type": "ineq", "fun": lambda x, A=A, b=b: b - A @ x},
        {"type": "eq", "fun": lambda x, Ai=Ai, bi=bi: Ai @ x - bi},
    ]

    res = linprog(
        c, A_ub=A, b_ub=b, A_eq=[Ai], b_eq=[bi], bounds=bounds, method="highs"
    )

    if res.success:
        return res.x
    return None


def find_step_limits(x0, d, A, b, Ai, bi, eps=1e-8):
    t_min, t_max = -np.inf, np.inf

    for j in range(len(b)):
        Aj = A[j]
        bj = b[j]

        denom = Aj @ d
        num = bj - Aj @ x0

        if abs(denom) < eps:
            if num < 0:
                return 0, 0  # No feasible step
            continue  # No constraint from this inequality

        t = num / denom
        if denom > 0:
            t_max = min(t_max, t)
        else:
            t_min = max(t_min, t)

    # Ensure we stay on the facet (A_i x = b_i)
    if abs(Ai @ d) > eps:
        t_eq = (bi - Ai @ x0) / (Ai @ d)
        t_min = max(t_min, t_eq - eps)
        t_max = min(t_max, t_eq + eps)

    return t_min, t_max


class ConvexPlaneComputer(nn.Module):
    def __init__(self, pointcloud, max_iters=50):
        super().__init__()
        self.max_iters = max_iters
        self.pointcloud = pointcloud
        device = self.pointcloud.device
        N, M = self.pointcloud.shape[0], targets.shape[0]
        self.unseparated_mask = torch.ones((M, N), dtype=torch.bool, device=device)

    def forward(self, targets):
        a_list, b_list = [], []
        self.unseparated_mask.fill_(True)
        for _ in range(self.max_iters):
            active = self.unseparated_mask.any(dim=1)
            if not active.any():
                break

            dists = torch.cdist(targets, self.pointcloud)  # (M, N)
            dists.masked_fill_(~self.unseparated_mask, float("inf"))

            closest_idx = dists.argmin(dim=1)  # (M,)
            closest_points = self.pointcloud[closest_idx]

            a_halfspace = closest_points - targets
            a_halfspace = a_halfspace / (a_halfspace.norm(dim=1, keepdim=True) + 1e-8)
            b_halfspace = torch.sum(a_halfspace * closest_points, dim=1)

            a_list.append(a_halfspace)
            b_list.append(b_halfspace)

            # Compute dot products efficiently
            dot = (
                torch.einsum("nd,md->mn", self.pointcloud, a_halfspace)
                - b_halfspace[:, None]
            )
            # dot = F.linear(self.pointcloud, a_halfspace, -b_halfspace)
            separated = dot >= -1e-6
            self.unseparated_mask.logical_and_(~separated)

        a_stack = torch.stack(a_list, dim=0)  # (I, M, 3)
        b_stack = torch.stack(b_list, dim=0)  # (I, M)

        return a_stack, b_stack


def compute_planes_triton(pointcloud, targets, max_iters=50):
    N, M = pointcloud.size(0), targets.size(0)
    mask = torch.ones((M, N), dtype=torch.bool, device=pointcloud.device)
    normals = torch.zeros((max_iters, M, 3), device=pointcloud.device)
    bs = torch.zeros((max_iters, M), device=pointcloud.device)

    for it in range(max_iters):
        grid = (triton.cdiv(M, 32), triton.cdiv(N, 128))
        iter_separation_kernel[grid](
            pointcloud,
            targets,
            mask,
            normals[it],
            bs[it],
            N,
            M,
            BLOCK_N=128,
            BLOCK_M=32,
        )
        if not mask.any():
            return normals[: it + 1], bs[: it + 1]

    return normals, bs


# Example: 2D square [0,1] x [0,1]
if __name__ == "__main__":
    A = np.array(
        [
            [1, 0, 0],
            [-1, 0, 0],
            [0, 1, 0],
            [0, -1, 0],
            [0, 0, 1],
            [0, 0, -1],
        ]
    )
    b = np.array([1, 1, 1, 1, 1, 1])

    samples = sample_boundary_points(A, b, num_samples=10000)

    device = "cuda"
    samples_th = torch.Tensor(samples).to(device)
    # targets = np.array([[0.5, 0.5, 0.5]] * 160)
    # targets_th = torch.Tensor(targets).to(device)
    targets_th = torch.randn((160, 3)).to(device)
    targets = targets_th.detach().cpu().numpy()

    model = ConvexPlaneComputer(samples_th, max_iters=50).cuda()
    compiled_model = model
    # compiled_model = torch.compile(model)

    # Warm-up run (important for accurate timing)
    _ = compiled_model(targets_th)

    # with profile(
    #     activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    #     schedule=torch.profiler.schedule(wait=0, warmup=1, active=2),
    #     on_trace_ready=torch.profiler.tensorboard_trace_handler("./log"),
    #     record_shapes=True,
    #     profile_memory=True,
    #     with_stack=True,
    # ) as prof:
    #     for _ in range(5):
    #         with record_function("forward_pass"):
    #             a_set, b_set = compiled_model(targets_th)
    #         prof.step()
    #
    # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))

    for i in range(2):
        t_start = time.perf_counter()
        a_set, b_set = compute_planes_triton(samples_th, targets_th)
        print(time.perf_counter() - t_start)
    a_set = a_set.detach().cpu().numpy().squeeze()
    b_set = b_set.detach().cpu().numpy().squeeze()

    # Visualize
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(samples[:, 0], samples[:, 1], samples[:, 2], s=5, color="blue")
    ax.set_title("Surface Samples from a 3D Cube")
    ax.set_box_aspect([1, 1, 1])
    for i in range(10):
        a_setc = np.concatenate((A, a_set[:, i, :]))
        b_setc = np.concatenate((b * 2, b_set[:, i]))
        plot_set(a_setc, b_setc)
        plt.plot(targets[i, 0], targets[i, 1], targets[i, 2], ".")
    plt.show()
