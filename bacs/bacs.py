import numpy as np
from scipy import linalg, sparse, stats

def skew(x):
    """Create skew-symmetric 3x3 matrix of 3x1 vector x"""
    return np.array([
        [0, -x[2, 0], x[1, 0]],
        [x[2, 0], 0, -x[0, 0]],
        [-x[1, 0], x[0, 0], 0],
    ])

def normS(x):
    """Spherically normalize n 3d vectors in form of a 3xn matrix"""
    return x / np.linalg.norm(x, axis=0)

def normS_jacobian(x):
    """Compute 3x3 Jacobian matrix for the spherical normalization of a 3d vector"""
    return (np.eye(3) - (x @ x.T) / (x.T @ x)) / np.linalg.norm(x)

def Rdr(dr):
    """Compute 3x3 rotation matrix from a small 3d rotation vector"""
    return np.linalg.solve(np.eye(3) - skew(dr), np.eye(3) + skew(dr))

def bacs(l, ijt, Sll, Xa, Ma, P, *,
         eps = 1e-6, # convergence criterion: max(abs(dl)) / sigma_l < eps
         max_iterations = 10, # maximum number of iterations
         tau = 0, # scale factor for Levenberg-Marquardt initialization
         k = np.inf, # threshold for Huber reweighting
         near_ratio = 1.0, # fraction of points used for datum definition
         sigma_h = (0, 0, 0, 0, 0, 0, 1e4)): # certainty of constraints

    # preprocess input for adjustment
    Ma_inv = np.stack([np.linalg.inv(ma) for ma in Ma], axis=0)
    Shh = np.diag(sigma_h)
    N = l.shape[1]
    I = Xa.shape[1]
    T = len(Ma)
    d = len(Shh)
    r = 2 * N - 3 * I - 6 * T + d
    if r < 0:
        raise ValueError(f'Not enough constraints (redundancy = {r})')
    qlsls = np.stack([normS_jacobian(l[:, n, None]) @ Sll[n] @ normS_jacobian(l[:, n, None]).T for n in range(N)], axis=2)
    l = normS(l)
    Xa = normS(Xa)
    la = normS(np.stack([P[ijt[n, 1]] @ Ma_inv[ijt[n, 2]] @ Xa[:, ijt[n, 0]] for n in range(N)], axis=1))
    w = np.ones((N, 1))
    W = sparse.identity(2 * N)
    beta = np.nan if np.isinf(k) else -2 * k * stats.norm.pdf(k) - 1 + 2 * stats.norm.cdf(k) * (1 - k**2) + 2 * k**2
    nu = 2

    # precompute indices for C, D and Q matrices
    c_rows = (2 * np.arange(N) + [[0], [1], [0], [1], [0], [1]]).flatten('F')
    c_cols = (3 * ijt[:, 0].T + [[0], [0], [1], [1], [2], [2]]).flatten('F')
    d_rows = (2 * np.arange(N) + [[0], [1], [0], [1], [0], [1], [0], [1], [0], [1], [0], [1]]).flatten('F')
    d_cols = (6 * ijt[:, 2].T + [[0], [0], [1], [1], [2], [2], [3], [3], [4], [4], [5], [5]]).flatten('F')
    q_rows = (2 * np.arange(N) + [[0], [1], [0], [1]]).flatten('F')
    q_cols = (2 * np.arange(N) + [[0], [0], [1], [1]]).flatten('F')

    # iterative estimation process
    for iteration in range(max_iterations):

        # nullspaces of object points
        nullXa = np.stack([linalg.null_space(Xa[:, i, None].T) for i in range(I)], axis=2)

        # Jacobians C and D, covariance matrix Qrr and reduced observations lr
        C = np.zeros((2, 3, N))
        D = np.zeros((2, 6, N))
        Qrr = np.zeros((2, 2, N))
        lr = np.zeros((2, N))
        for n in range(N):
            i, j, t = ijt[n, :]
            nullLa = linalg.null_space(la[:, n, None].T)
            tmp = nullLa.T @ normS_jacobian(P[j] @ Ma_inv[t] @ Xa[:, i, None]) @ P[j]
            C[:, :, n] = tmp @ Ma_inv[t] @ nullXa[:, :, i]
            D[:, :, n] = tmp[:, :3] @ np.hstack((-skew(Ma_inv[t, :3, :] @ Xa[:, i, None]), Xa[3, i] * np.eye(3)))
            Qrr[:, :, n] = nullLa.T @ qlsls[:, :, n] @ nullLa
            lr[:, n] = nullLa.T @ l[:, n]
        C = sparse.csc_matrix((C.flatten('F'), (c_rows, c_cols)))
        D = sparse.csc_matrix((D.flatten('F'), (d_rows, d_cols)))
        invQrr = np.vstack((
            np.hstack((Qrr[1:, 1:, :], -Qrr[:1, 1:, :])),
            np.hstack((-Qrr[:1, 1:, :], Qrr[:1, :1, :])),
        )) / (Qrr[:1, :1, :] * Qrr[1:, 1:, :] - Qrr[:1, 1:, :]**2)
        invQrr = sparse.csc_matrix((invQrr.flatten('F'), (q_rows, q_cols))) @ W

        # linear centroid constraints for datum definition
        n_fix = round(near_ratio * I)
        indices = np.argsort(Xa[-1, :])[::-1]
        indices = np.sort(indices[:n_fix])
        Xa_fix = Xa[:, indices]
        H_fix = np.vstack([
            nullXa[:, :, indices[i]].T @
            (1 / Xa_fix[3, i]**2 * np.hstack((Xa_fix[3, i] * np.eye(3), -Xa_fix[:3, i, None])).T) @ \
            np.hstack((np.eye(3), -skew(Xa_fix[:3, i, None] / Xa_fix[3, i]), Xa_fix[:3, i, None] / Xa_fix[3, i]))
            for i in range(n_fix)
        ])
        H = np.zeros((3 * I, d))
        H[(3 * indices + [[0], [1], [2]]).flatten('F'), :] = H_fix
        Nside = np.vstack((H, np.zeros((6 * T, d))))

        # parameter and observation updates
        A = sparse.hstack((C, D))
        Nmat = sparse.vstack((
            sparse.hstack((A.T @ invQrr @ A, Nside)),
            np.hstack((Nside.T, -Shh)),
        ))
        nvec = np.vstack((
            (A.T @ invQrr @ lr.flatten('F'))[:, None],
            np.zeros((d, 1)),
        ))

        # Levenberg-Marquardt
        if iteration == 0:
            mu = tau * np.sqrt(np.max(Nmat.diagonal()))
        dx = sparse.linalg.spsolve(Nmat + mu * sparse.identity(Nmat.shape[0]), nvec)

        # parameter update
        dk = dx[:3*I, None]
        Xa = normS(np.hstack([
            Xa[:, i, None] + nullXa[:, :, i] @ dk[3*i:3*i+3]
            for i in range(I)
        ]))
        dd = dx[3*I:-d, None]
        Ma_inv = np.stack([
            np.vstack((
                np.hstack((Rdr(dd[6*t:6*t+3] / 2), dd[6*t+3:6*t+6])),
                np.array([0, 0, 0, 1]),
            )) @ Ma_inv[t]
            for t in range(T)
        ], axis=0)

        # observation update
        dl = A @ dx[:-d]

        # approximate observations for next iteration
        la = normS(np.stack([
            P[ijt[n, 1]] @ Ma_inv[ijt[n, 2]] @ Xa[:, ijt[n, 0]]
            for n in range(N)
        ], axis=1))

        # non-linear corrections
        vr = np.stack([
            -linalg.null_space(la[:, n, None].T).T @ l[:, n]
            for n in range(N)
        ], axis=1)

        # check for convergence
        criterion = np.max(np.abs(dl) * np.sqrt(invQrr.diagonal()))
        reweighted = np.sum(w < 1) / N
        print(f'Iteration {iteration+1} of {max_iterations}:',
              f'mu = {mu},',
              f'convergence = {criterion:.0e},',
              f'{reweighted*100:.2f}% reweighted')
        if criterion <= eps:
            if mu == 0:
                break
            mu = 0
        else:
            # Levenberg-Marquardt
            phi = (lr.flatten('F')[:, None].T @ invQrr @ lr.flatten('F')[:, None] -
                   vr.flatten('F')[:, None].T @ invQrr @ vr.flatten('F')[:, None]) / \
                  (np.vstack((dk, dd)).T @ (mu * np.vstack((dk, dd)) + A.T @ invQrr @ lr.flatten('F')[:, None]))
            if phi > 0:
                mu = mu * max(1 / 3, 1 - (2 * phi[0, 0] - 1)**3)
                nu = 2
            else:
                mu = mu * nu
                nu = 2 * nu

        # robust variance factor for reweighting
        diag_Qrr = np.stack((Qrr[0, 0, :], Qrr[1, 1, :]), axis=1).flatten()
        res = np.abs(vr.flatten('F')) / np.sqrt(diag_Qrr)
        res[res > k] = k
        s0 = np.linalg.norm(res) / np.sqrt(r * beta)

        # pointwise reweighting according to corrections in tangent space
        w = np.ones((N, 1))
        res = np.abs(vr.flatten('F')) / np.sqrt(diag_Qrr)
        res = np.sqrt(np.sum(res.reshape(-1, 2), axis=1))
        btk = res > k * s0
        if np.any(btk):
            w[btk] = s0 * k * w[btk] / res[btk, None]
            W = sparse.diags(w.repeat(2))

    # correct direction of homogeneus scene points at the horizon
    Xa[-1, Xa[-1, :] < 0] = -Xa[-1, Xa[-1, :] < 0]

    # invert for motion matrix from world to mcs
    Ma = np.stack([np.linalg.inv(ma_inv) for ma_inv in Ma_inv], axis=0)

    # empirical covariance matrix
    if r > 0:
        s0dsq = vr.flatten('F')[:, None].T @ invQrr @ vr.flatten('F')[:, None] / r
        f = 1 if r < 30 else s0dsq[0, 0]
    else:
        s0dsq = np.nan
    Ncsc = Nmat.tocsc()
    N00 = Ncsc[:3*I, :3*I]
    N01 = Ncsc[:3*I, 3*I:]
    N10 = Ncsc[3*I:, :3*I]
    N11 = Ncsc[3*I:, 3*I:]
    Sdd = f * sparse.linalg.inv(N11 - N10 @ sparse.linalg.spsolve(N00, N01))
    Sdd = Sdd[:6*T, :6*T]

    return la, Xa, Ma, Sdd, s0dsq, vr, w, iteration