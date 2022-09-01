import os

import numpy as np


def read_input(dataset: str):

    def load(filename: str):
        return np.loadtxt(os.path.join('input', dataset, filename), delimiter=',')

    l = load('rays.dat').T
    Sll = np.array([[
        [line[0], line[3], line[5]],
        [line[3], line[1], line[4]],
        [line[5], line[4], line[2]],
    ] for line in load('covariances.dat')])
    ijt = load('linkage.dat').astype(int) - 1  # NOTE: from base-1 to base-0
    Xa = load('points.dat').T
    Ma = load('motions.dat').reshape(-1, 4, 4)
    P = load('projections.dat').reshape(-1, 3, 4)

    return l, Sll, ijt, Xa, Ma, P


def write_output(dataset: str, ld, Xd, Md, Sdd, vr, w):

    def save(filename: str, X, fmt: str = '%.6f'):
        np.savetxt(os.path.join('results', dataset, filename), X, delimiter=',', fmt=fmt)

    save('rays.dat', ld.T)
    save('points.dat', Xd.T)
    save('motions.dat', Md.reshape(-1, 4))
    save('motioncovariance.dat', Sdd.toarray(), '%.12f')
    save('corrections.dat', vr.T, '%.12f')
    save('weights.dat', w)
