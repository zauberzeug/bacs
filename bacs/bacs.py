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

def bacs(l, ict, Sll, Xa, Ma, P, *,
         eps = 1e-6,
         max_iterations = 10,
         tau = 0,
         k = np.inf,
         near_ratio = 1.0,
         sigma_h = (0, 0, 0, 0, 0, 0, 100),
         sigma_c = None,
         ):
    """
    Perform a bundle adjustment for multi-view cameras based on
    corresponding camera rays.

    On the supposition that the deviations of the observed camera rays are
    normally distributed and mutually independent, the estimated orientation
    parameters and object point coordinates correspond to the maximum-
    likelihood estimation.
    Line preserving cameras with known inner calibration as well as their
    orientation within the multi-camera system (MCS) are assumed to be known.
    For initialization sufficiently accurate approximate values for scene
    point coordinates and orientations of the MCS (translation and rotation)
    for different instances of time are needed.

    Dimensions:
        N  number of observed camera rays
        I  number of object points
        C  number of cameras within the MCS
        T  number of instances in time

    Input:
        l    camera rays [3xN]
        ict  observation linkage [Nx3], n-th row contains [i,c,t]
        Sll  covariance matrices [Nx3x3] of each camera ray l[:,n]
        Xa   approximate values for homogeneous object points [4xI]
        Ma   approximate values for MCS transformations [Tx4x4]
        P    projection matrices of the single-view cams [Cx3x4]
        eps  convergence criterion: max(abs(dl)) / sigma_l < eps (default: 1e-6)
        max_iterations   maximum number of iterations (default: 10)
        tau  scale factor for Levenberg-Marquardt initialization (default: 0 -> no Levenberg-Marquardt)
        k    threshold for Huber reweighting (default: Inf -> no reweighting)
        near_ratio  fraction of scene points used for datum definition (default: 1 -> all)
        sigma_h  variances of centroid constraints (3 translations, 3 rotations, 1 scale, default: (0,0,0,0,0,0,100))
        sigma_c  variances of camera motion constraints (experimental, length 6xT, default: None)

    Output:
        la   estimated camera rays [3xN]
        Xa   estimated object point coordinates (homogeneous) [4xN]
        Ma   estimated motion matrices [Tx4x4] (from object system to MCS)
        Sdd  estimated covariance matrix of oriantation parameters [6Tx6T]
        s0dsq  estimated variance factor
        vr   estimated corrections on l in tangent space
        w    estimated weights on diagonal elements of Sll
        iterations  number of iterations
    """

    # preprocess input for adjustment
    Ma_inv = np.stack([np.linalg.inv(ma) for ma in Ma], axis=0)
    N = l.shape[1]
    I = Xa.shape[1]
    T = len(Ma)
    c_indices = [k for k, s in enumerate(sigma_c) if s is not None and not np.isinf(s)] if sigma_c is not None else [] 
    sigma_c = [s for s in sigma_c if s is not None and not np.isinf(s)] if sigma_c is not None else []
    Shh = sparse.diags(list(sigma_h) + sigma_c)**2
    d = Shh.shape[0]
    r = 2 * N - 3 * I - 6 * T + d
    if r < 0:
        raise ValueError(f'Not enough constraints (redundancy = {r})')
    qlsls = np.stack([normS_jacobian(l[:, n, None]) @ Sll[n] @ normS_jacobian(l[:, n, None]).T for n in range(N)], axis=2)
    l = normS(l)
    Xa = normS(Xa)
    la = normS(np.stack([P[ict[n, 1]] @ Ma_inv[ict[n, 2]] @ Xa[:, ict[n, 0]] for n in range(N)], axis=1))
    w = np.ones((N, 1))
    W = sparse.identity(2 * N)
    beta = np.nan if np.isinf(k) else -2 * k * stats.norm.pdf(k) - 1 + 2 * stats.norm.cdf(k) * (1 - k**2) + 2 * k**2
    nu = 2

    # precompute indices for C, D and Q matrices
    c_rows = (2 * np.arange(N) + [[0], [1], [0], [1], [0], [1]]).flatten('F')
    c_cols = (3 * ict[:, 0].T + [[0], [0], [1], [1], [2], [2]]).flatten('F')
    d_rows = (2 * np.arange(N) + [[0], [1], [0], [1], [0], [1], [0], [1], [0], [1], [0], [1]]).flatten('F')
    d_cols = (6 * ict[:, 2].T + [[0], [0], [1], [1], [2], [2], [3], [3], [4], [4], [5], [5]]).flatten('F')
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
            i, c, t = ict[n, :]
            nullLa = linalg.null_space(la[:, n, None].T)
            tmp = nullLa.T @ normS_jacobian(P[c] @ Ma_inv[t] @ Xa[:, i, None]) @ P[c]
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
        Nside = np.zeros((3 * I + 6 * T, d))
        Nside[(3 * indices + [[0], [1], [2]]).flatten('F'), :7] = H_fix
        for k, index in enumerate(c_indices):
            Nside[3 * I + index, 7 + k] = 1

        # parameter and observation updates
        A = sparse.hstack((C, D))
        Nmat = sparse.vstack((
            sparse.hstack((A.T @ invQrr @ A, Nside)),
            sparse.hstack((Nside.T, -Shh)),
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
            P[ict[n, 1]] @ Ma_inv[ict[n, 2]] @ Xa[:, ict[n, 0]]
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
        s0dsq = (vr.flatten('F')[:, None].T @ invQrr @ vr.flatten('F')[:, None] / r)[0, 0]
        f = 1 if r < 30 else s0dsq
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