# -*- coding: utf-8 -*-
"""
Python implemention of PPCA-EM for data with missing values
Adapted from MATLAB implemention from J.J. VerBeek
"""

import numpy as np
from numpy import shape, isnan, nanmean, average, log, cov
from numpy.matlib import repmat
from numpy.random import normal
from numpy.linalg import inv, det, eig
from numpy import identity as eye
from numpy import trace as tr
from scipy.linalg import orth


def PPCA(Y, d):
    """
    Implements probabilistic PCA for data with missing values,
    using a factorizing distribution over hidden states and hidden observations.

    Args:
        Y (np.ndarray) input numpy ndarray of data vectors (N  by D)
        d (int): dimension of latent space

    Returns:
        C (D by d): C*C' + I*ss is covariance model, C has scaled principal directions as cols
        ss (float): isotropic variance outside subspace
        M (D by 1): data mean
        X (N by d): expected states
        Ye (N by D): expected complete observations (differs from Y if data is missing)

        Based on MATLAB code from J.J. VerBeek, 2006. http://lear.inrialpes.fr/~verbeek
    """
    N, D = shape(Y)  # N observations in D dimensions (i.e. D is number of features, N is samples)
    threshold = 1E-4  # minimal relative change in objective function to continue
    hidden = isnan(Y)
    missing = hidden.sum()

    if missing > 0:
        M = nanmean(Y, axis=0)
    else:
        M = average(Y, axis=0)

    Ye = Y - repmat(M, N, 1)

    if missing > 0:
        Ye[hidden] = 0

    # initialize
    C = normal(loc=0.0, scale=1.0, size=(D, d))
    CtC = C.T @ C
    X = Ye @ C @ inv(CtC)
    recon = X @ C.T
    recon[hidden] = 0
    ss = np.sum((recon - Ye) ** 2) / (N * D - missing)

    count = 1
    old = np.inf

    # EM Iterations
    while (count):
        Sx = inv(eye(d) + CtC / ss)  # E-step, covariances
        ss_old = ss
        if missing > 0:
            proj = X @ C.T
            Ye[hidden] = proj[hidden]

        X = Ye @ C @ Sx / ss  # E-step: expected values

        SumXtX = X.T @ X  # M-step
        C = Ye.T @ X @ (SumXtX + N * Sx).T @ inv(((SumXtX + N * Sx) @ (SumXtX + N * Sx).T))
        CtC = C.T @ C
        ss = (np.sum((X @ C.T - Ye) ** 2) + N * np.sum(CtC * Sx) + missing * ss_old) / (N * D)
        # transform Sx determinant into numpy longdouble in order to deal with high dimensionality
        Sx_det = np.min(Sx).astype(np.longdouble) ** shape(Sx)[0] * det(Sx / np.min(Sx))
        objective = N * D + N * (D * log(ss) + tr(Sx) - log(Sx_det)) + tr(SumXtX) - missing * log(ss_old)

        rel_ch = np.abs(1 - objective / old)
        old = objective

        count = count + 1
        if rel_ch < threshold and count > 5:
            count = 0

    C = orth(C)
    covM = cov((Ye @ C).T)
    if d == 1:
        covM = [[covM]]
    vals, vecs = eig(covM)
    ordr = np.argsort(vals)[::-1]
    vecs = vecs[:, ordr]

    C = C @ vecs
    X = Ye @ C

    # add data mean to expected complete data
    Ye = Ye + repmat(M, N, 1)

    return C, ss, M, X, Ye
