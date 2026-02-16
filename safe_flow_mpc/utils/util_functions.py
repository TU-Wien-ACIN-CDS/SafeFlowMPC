from collections import namedtuple

import cdd
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation as R


def make_box(lb, ub):
    a_set = np.concatenate((np.eye(3), -np.eye(3)))
    b_set = np.concatenate((ub, -np.array(lb)))
    return [a_set, b_set]


def compute_polytope_vertices(a_set, b_set):
    b_set = b_set.reshape((b_set.shape[0], 1))
    array = np.hstack([b_set, -a_set])
    mat = cdd.matrix_from_array(array, rep_type=cdd.RepType.INEQUALITY)
    poly = cdd.polyhedron_from_matrix(mat)
    g = cdd.copy_generators(poly)
    V = np.array(g.array)
    vertices = []
    for i in range(V.shape[0]):
        if V[i, 0] != 1:  # 1 = vertex, 0 = ray
            raise ValueError("Polyhedron is not a polytope")
        elif i not in g.lin_set:
            vertices.append(V[i, 1:])
    return vertices


def compute_polytope_edges(vertices):
    hull = ConvexHull(vertices)
    edges = []
    for simplex in hull.simplices:
        for i in range(len(simplex)):
            for j in range(i + 1, len(simplex)):
                edge = sorted([simplex[i], simplex[j]])
                if edge not in edges:
                    edges.append(edge)
    return edges


def reduce_ineqs(a_set, b_set):
    b_set = b_set.reshape((b_set.shape[0], 1))
    array = np.hstack([b_set, -a_set])
    mat = cdd.matrix_from_array(array, rep_type=cdd.RepType.INEQUALITY)
    cdd.matrix_redundancy_remove(mat)
    mat_np = np.array(mat.array)
    return [-mat_np[:, 1:], mat_np[:, 0]]


def plot_set(a_set, b_set, color=0):
    points = np.array(compute_polytope_vertices(a_set, b_set))
    hull = ConvexHull(points)
    faces = hull.simplices
    for face in faces:
        p1, p2, p3 = np.array(points)[face]
        dps = [[p1, p2], [p1, p3], [p2, p3]]
        for dp in dps:
            plt.plot(
                [dp[0][0], dp[1][0]],
                [dp[0][1], dp[1][1]],
                [dp[0][2], dp[1][2]],
                f"C{color}",
            )
    plt.axis("equal")


def plot_set_2d(a_set, b_set, color=0):
    points = np.array(compute_polytope_vertices(a_set, b_set))
    hull = ConvexHull(points)
    faces = hull.simplices
    for face in faces:
        p1, p2 = np.array(points)[face]
        dps = [[p1, p2]]
        for dp in dps:
            plt.plot(
                [dp[0][0], dp[1][0]],
                [dp[0][1], dp[1][1]],
                f"C{color}",
            )
    plt.axis("equal")


def normalize_set_size(sets, max_set_size=30):
    for set_iter in sets:
        a_norm = np.zeros((max_set_size, 3))
        b_norm = 1 * np.ones(max_set_size)
        set_size = set_iter[0].shape[0]
        if set_size <= max_set_size:
            a_norm[:set_size, :] = set_iter[0]
            b_norm[:set_size] = set_iter[1]
            set_iter[0] = a_norm
            set_iter[1] = b_norm
        else:
            print(
                f"(SetNormalizer) ERROR set size {set_size} exceeds max set size {max_set_size}"
            )
    return sets
