import matplotlib.pyplot as plt
import numpy as np

from odl.util import many_matvec, many_matmul
from odl.util.graphics import warning_free_pause

# %%  Definitions, change to test various settings

ell1_center = [1, 1]
ell1_vectors = [[1, 1], [-1, 1]]
ell1_halfaxes = [1, 2]
ell2_center = [-1, -1]
ell2_vectors = [[1, 0], [0, 1]]
ell2_halfaxes = [2, 1]

# %% Preparation code

c1 = np.asarray(ell1_center, dtype=float)
U1T = np.asarray(ell1_vectors, dtype=float)
e1 = np.asarray(ell1_halfaxes, dtype=float)
c2 = np.asarray(ell2_center, dtype=float)
U2T = np.asarray(ell2_vectors, dtype=float)
e2 = np.asarray(ell2_halfaxes, dtype=float)

d = c1.size
if c2.shape == (d,):
    c2 = c2[None, :]
    U2T = U2T[None, :, :]
    e2 = e2[None, :]

# --- Re-normalize and transpose the U matrices --- #

U1T = U1T / np.linalg.norm(U1T, axis=1, keepdims=True)
U1T = U1T[None, :, :]
U1 = np.transpose(U1T, (0, 2, 1))
U2T = U2T / np.linalg.norm(U2T, axis=2, keepdims=True)
U2 = np.transpose(U2T, (0, 2, 1))

# Add empty axes for the vectors to match the "many" type arrays of
# ellipsoid 2
c1 = c1[None, :]
e1 = e1[None, :]

# --- Compute level set representation parts --- #

A1 = many_matmul(U1 / e1[:, None, :] ** 2, U1T)
b1 = -many_matvec(A1, c1)
A2 = many_matmul(U2 / e2[:, None, :] ** 2, U2T)
b2 = -many_matvec(A2, c2)

gamma1 = 1 / np.linalg.norm(A1, float('inf'), axis=(1, 2))
gamma2 = 1 / np.linalg.norm(A2, float('inf'), axis=(1, 2))

cent1, cent2 = c1, c2  # for backup


# --- Define helpers --- #

def angle(u, v):
    """Return the angle between u and v (many vectors)."""
    norm_u = np.linalg.norm(u, axis=1)
    norm_v = np.linalg.norm(v, axis=1)
    nonzero = (norm_u > 1e-10) & (norm_v > 1e-10)
    ang = np.zeros(u.shape[0])
    inner = np.zeros_like(ang)
    inner[nonzero] = (
        np.tensordot(u[nonzero], v[nonzero], axes=[1, 1]) /
        (norm_u[nonzero] * norm_v[nonzero])
    ).ravel()
    ang[nonzero] = np.arccos(np.clip(inner, -1, 1))
    return ang


def find_t_in_0_1(u, v):
    r"""Helper to find a t value for the iteration.

    This function solves the quadratic equation

    .. math::
        \|v\|^2\, t^2 + 2 u^{\mathrm{T}}v\, t + (\|u\|^2 - 1) = 0

    for :math:`t \in [0, 1]`. Due to the way the vectors :math:`u`
    and :math:`v` are defined, such a solution always exists.

    The two possible solutions are

    .. math::
        t_\pm = \frac{
            \pm \sqrt{(u^{\mathrm{T}}v)^2 - \|v\|^2(\|u\|^2 - 1)} -
            u^{\mathrm{T}}v
            }{
                \|v\|^2
            }.
    """
    u_norm2 = np.linalg.norm(u, axis=1) ** 2
    v_norm2 = np.linalg.norm(v, axis=1) ** 2
    u_dot_v = np.tensordot(u, v, axes=[1, 1]).ravel()
    sqrt = np.sqrt(u_dot_v ** 2 - v_norm2 * (u_norm2 - 1))
    tplus = (sqrt - u_dot_v) / v_norm2
    tminus = (-sqrt - u_dot_v) / v_norm2
    # Make sure that not finding a root in [0, 1] breaks subsequent code
    t = np.full_like(tplus, fill_value=float('nan'))
    plus = (tplus >= 0) & (tplus <= 1)
    minus = (tminus >= 0) & (tminus <= 1)
    t[plus] = tplus[plus]
    t[minus] = tminus[minus]
    return t


def ell1_curve(phi):
    """Return points on the boundary of the first ellipse."""
    phi = np.array(phi, ndmin=1, copy=False)
    u0 = ell1_vectors[0] / np.linalg.norm(ell1_vectors[0])
    u1 = ell1_vectors[1] / np.linalg.norm(ell1_vectors[1])
    a, b = ell1_halfaxes

    return np.array(ell1_center) + (
        np.cos(phi)[:, None] * (a * u0)[None, :] +
        np.sin(phi)[:, None] * (b * u1)[None, :]
    )


def ell2_curve(phi):
    """Return points on the boundary of the second ellipse."""
    phi = np.array(phi, ndmin=1, copy=False)
    u0 = ell2_vectors[0] / np.linalg.norm(ell2_vectors[0])
    u1 = ell2_vectors[1] / np.linalg.norm(ell2_vectors[1])
    a, b = ell2_halfaxes

    return np.array(ell2_center) + (
        np.cos(phi)[:, None] * (a * u0)[None, :] +
        np.sin(phi)[:, None] * (b * u1)[None, :]
    )


# %% Make iteration plot

niter = 10
phi = np.linspace(0, 2 * np.pi, 181)
ell1 = ell1_curve(phi)
ell2 = ell2_curve(phi)

fig, ax = plt.subplots()
p_ell1 = ax.plot(ell1[:, 0], ell1[:, 1], color='blue')
p_ell2 = ax.plot(ell2[:, 0], ell2[:, 1], color='green')
p_cent1 = ax.scatter(cent1.T[0], cent1.T[1], c='blue')
p_cent2 = ax.scatter(cent2.T[0], cent2.T[1], c='green')
warning_free_pause(2)

# --- Iteration 0, define plot things --- #

print('Iteration 0')
u1 = many_matvec(U1T, c1 - cent1) / e1
v1 = many_matvec(U1T, c2 - c1) / e1
t1 = find_t_in_0_1(u1, v1)
u2 = many_matvec(U2T, c1 - cent2) / e2
v2 = many_matvec(U2T, c2 - c1) / e2
t2 = find_t_in_0_1(u2, v2)
z1 = c1 + t1 * (c2 - c1)
z2 = c1 + t2 * (c2 - c1)
normal1 = many_matvec(A1, z1) + b1
normal2 = many_matvec(A2, z2) + b2

p_c1 = ax.scatter(c1.T[0], c1.T[1], c='blue')
p_c2 = ax.scatter(c2.T[0], c2.T[1], c='green')
p_z1 = ax.scatter(z1.T[0], z1.T[1], c='blue')
p_z2 = ax.scatter(z2.T[0], z2.T[1], c='green')
warning_free_pause(0.5)
p_normal1 = ax.arrow(z1[0][0], z1[0][1], normal1[0][0], normal1[0][1],
                     color='blue', head_width=0.1)
p_normal2 = ax.arrow(z2[0][0], z2[0][1], normal2[0][0], normal2[0][1],
                     color='green', head_width=0.1)
warning_free_pause(1.5)

# --- Iteration k, update plot in-place --- #

for i in range(niter):
    print('Iteration {}'.format(i + 1))
    c1 = z1 - gamma1 * normal1
    c2 = z2 - gamma2 * normal2
    u1 = many_matvec(U1T, c1 - cent1) / e1
    v1 = many_matvec(U1T, c2 - c1) / e1
    t1 = find_t_in_0_1(u1, v1)
    u2 = many_matvec(U2T, c1 - cent2) / e2
    v2 = many_matvec(U2T, c2 - c1) / e2
    t2 = find_t_in_0_1(u2, v2)
    z1 = c1 + t1 * (c2 - c1)
    z2 = c1 + t2 * (c2 - c1)
    normal1 = many_matvec(A1, z1) + b1
    normal2 = many_matvec(A2, z2) + b2

    p_c1.set_offsets(c1)
    p_c2.set_offsets(c2)
    warning_free_pause(0.5)
    p_z1.set_offsets(z1)
    p_z2.set_offsets(z2)
    warning_free_pause(0.5)
    ax.artists.remove(p_normal1)
    ax.artists.remove(p_normal2)
    p_normal1 = ax.arrow(z1[0][0], z1[0][1], normal1[0][0], normal1[0][1],
                         color='blue', head_width=0.1)
    p_normal2 = ax.arrow(z2[0][0], z2[0][1], normal2[0][0], normal2[0][1],
                         color='green', head_width=0.1)
    warning_free_pause(1)
