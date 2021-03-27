from __future__ import division
import math
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
from matplotlib import colors
import matplotlib.pyplot as plt
from time import sleep
import pandas as pd
import numpy as np

import cupy as cp
import timeit
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
from matplotlib import colors


def main(nelx, nely, volfrac, penal, rmin, ft, floc, fx, fy, bcloc, fout):
    myGlobalStr = ""
    print("Minimum compliance problem with OC")
    print("ndes: " + str(nelx) + " x " + str(nely))
    print(
        "volfrac: " + str(volfrac) + ", rmin: " + str(rmin) + ", penal: " + str(penal)
    )
    print("Filter method: " + ["Sensitivity based", "Density based"][ft])
    # Max and min stiffness
    Emin = 1e-9
    Emax = 1.0
    # dofs:
    ndof = 2 * (nelx + 1) * (nely + 1)
    # Allocate design variables (as array), initialize and allocate sens.
    x = volfrac * np.ones(nely * nelx, dtype=float)
    xold = x.copy()
    xPhys = x.copy()
    g = 0  # must be initialized to use the NGuyen/Paulino OC approach
    dc = np.zeros((nely, nelx), dtype=float)
    # FE: Build the index vectors for the for coo matrix format.
    KE = lk()
    edofMat = np.zeros((nelx * nely, 8), dtype=int)
    for elx in range(nelx):
        for ely in range(nely):
            el = ely + elx * nely
            n1 = (nely + 1) * elx + ely
            n2 = (nely + 1) * (elx + 1) + ely
            edofMat[el, :] = np.array(
                [
                    2 * n1 + 2,
                    2 * n1 + 3,
                    2 * n2 + 2,
                    2 * n2 + 3,
                    2 * n2,
                    2 * n2 + 1,
                    2 * n1,
                    2 * n1 + 1,
                ]
            )
    # Construct the index pointers for the coo format
    iK = np.kron(edofMat, np.ones((8, 1))).flatten()
    jK = np.kron(edofMat, np.ones((1, 8))).flatten()
    # Filter: Build (and assemble) the index+data vectors for the coo matrix format
    nfilter = int(nelx * nely * ((2 * (np.ceil(rmin) - 1) + 1) ** 2))
    iH = np.zeros(nfilter)
    jH = np.zeros(nfilter)
    sH = np.zeros(nfilter)
    cc = 0
    for i in range(nelx):
        for j in range(nely):
            row = i * nely + j
            kk1 = int(np.maximum(i - (np.ceil(rmin) - 1), 0))
            kk2 = int(np.minimum(i + np.ceil(rmin), nelx))
            ll1 = int(np.maximum(j - (np.ceil(rmin) - 1), 0))
            ll2 = int(np.minimum(j + np.ceil(rmin), nely))
            for k in range(kk1, kk2):
                for l in range(ll1, ll2):
                    col = k * nely + l
                    fac = rmin - np.sqrt(((i - k) * (i - k) + (j - l) * (j - l)))
                    iH[cc] = row
                    jH[cc] = col
                    sH[cc] = np.maximum(0.0, fac)
                    cc = cc + 1
    # Finalize assembly and convert to csc format
    H = coo_matrix((sH, (iH, jH)), shape=(nelx * nely, nelx * nely)).tocsc()
    Hs = H.sum(1)
    # BC's and support
    dofs = np.arange(2 * (nelx + 1) * (nely + 1))
    fixed = []
    if bcloc == "l":
        for i in range(nely + 1):
            ind = 2 * i
            fixed.append(dofs[ind])
            fixed.append(dofs[ind + 1])
    elif bcloc == "r":
        for i in range(nely + 1):
            ind = 2 * i + 2 * (nelx) * (nely + 1)
            fixed.append(dofs[ind])
            fixed.append(dofs[ind + 1])
    elif bcloc == "t":
        for i in range(nelx + 1):
            ind = 2 * i * (nely + 1)
            fixed.append(dofs[ind])
            fixed.append(dofs[ind + 1])
    elif bcloc == "b":
        for i in range(nelx + 1):
            ind = (nely + 1) + 2 * i * (nely + 1)
            fixed.append(dofs[ind])
            fixed.append(dofs[ind + 1])
    else:
        fixed = np.union1d(
            dofs[0 : 2 * (nely + 1) : 2], np.array([2 * (nelx + 1) * (nely + 1) - 1])
        )
    print("new fixed = ", fixed)

    free = np.setdiff1d(dofs, fixed)
    # Solution and RHS vectors
    f = np.zeros((ndof, 1))
    u = np.zeros((ndof, 1))
    # Set load
    if floc == "tr":
        f[2 * (nelx) * (nely + 1), 0] = fx  # tr
        f[2 * (nelx) * (nely + 1) + 1, 0] = fy  # tr
    elif floc == "br":
        f[2 * (nelx + 1) * (nely + 1) - 2, 0] = fx  # br
        f[2 * (nelx + 1) * (nely + 1) - 1, 0] = fy  # br
    elif floc == "tl":
        f[0, 0] = fx  # tl
        f[1, 0] = fy  # tl
    elif floc == "bl":
        f[2 * (nely + 1) - 2, 0] = fx  # bl
        f[2 * (nely + 1) - 1, 0] = fy  # bl
    else:
        f[1, 0] = -1  # tl
    # Initialize plot and plot the initial design
    # Set loop counter and gradient vectors
    loop = 0
    change = 1
    dv = np.ones(nely * nelx)
    dc = np.ones(nely * nelx)
    ce = np.ones(nely * nelx)
    TotalLoops = 100
    nPrint = 20
    PrintFreq = int(TotalLoops / nPrint)
    print("PrintFreq = ", PrintFreq)
    while change > 0.001 and loop < TotalLoops:
        loop = loop + 1
        # Setup and solve FE problem
        sK = (
            (KE.flatten()[np.newaxis]).T * (Emin + (xPhys) ** penal * (Emax - Emin))
        ).flatten(order="F")
        K = coo_matrix((sK, (iK, jK)), shape=(ndof, ndof)).tocsc()
        # Remove constrained dofs from matrix
        K = K[free, :][:, free]
        # Solve system
        u[free, 0] = spsolve(K, f[free, 0])
        # Objective and sensitivity
        ce[:] = (
            np.dot(u[edofMat].reshape(nelx * nely, 8), KE)
            * u[edofMat].reshape(nelx * nely, 8)
        ).sum(1)
        obj = ((Emin + xPhys ** penal * (Emax - Emin)) * ce).sum()
        dc[:] = (-penal * xPhys ** (penal - 1) * (Emax - Emin)) * ce
        dv[:] = np.ones(nely * nelx)
        # Sensitivity filtering:
        if ft == 0:
            dc[:] = np.asarray((H * (x * dc))[np.newaxis].T / Hs)[:, 0] / np.maximum(
                0.001, x
            )
        elif ft == 1:
            dc[:] = np.asarray(H * (dc[np.newaxis].T / Hs))[:, 0]
            dv[:] = np.asarray(H * (dv[np.newaxis].T / Hs))[:, 0]
        # Optimality criteria
        xold[:] = x
        (x[:], g) = oc(nelx, nely, x, volfrac, dc, dv, g)
        # Filter design variables
        if ft == 0:
            xPhys[:] = x
        elif ft == 1:
            xPhys[:] = np.asarray(H * x[np.newaxis].T / Hs)[:, 0]
        # Compute the change by the inf. norm
        change = np.linalg.norm(
            x.reshape(nelx * nely, 1) - xold.reshape(nelx * nely, 1), np.inf
        )
        # Write iteration history to screen (req. Python 2.6 or newer)
        print(
            "it.: {0} , obj.: {1:.3f} Vol.: {2:.3f}, ch.: {3:.3f}".format(
                loop, obj, (g + volfrac * nelx * nely) / (nelx * nely), change
            )
        )
        objVal = "%.2f" % obj
        volVal = "%.2f" % ((g + volfrac * nelx * nely) / (nelx * nely))
        changeVal = "%.2f" % change
        myGlobalStr = "iteration : {it} , \t objective: {obj}, \t Volume fraction: {vol}, \t change in volume: {change}".format(
            it=loop, obj=objVal, vol=volVal, change=changeVal
        )
        if loop % PrintFreq == 0:
            print(myGlobalStr, file=fout, flush=True)

    return xPhys


# element stiffness matrix
def lk():
    E = 1
    nu = 0.3
    k = np.array(
        [
            1 / 2 - nu / 6,
            1 / 8 + nu / 8,
            -1 / 4 - nu / 12,
            -1 / 8 + 3 * nu / 8,
            -1 / 4 + nu / 12,
            -1 / 8 - nu / 8,
            nu / 6,
            1 / 8 - 3 * nu / 8,
        ]
    )
    KE = (
        E
        / (1 - nu ** 2)
        * np.array(
            [
                [k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]],
                [k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2]],
                [k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1]],
                [k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4]],
                [k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3]],
                [k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6]],
                [k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5]],
                [k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]],
            ]
        )
    )
    return KE


# Optimality criterion
def oc(nelx, nely, x, volfrac, dc, dv, g):
    l1 = 0
    l2 = 1e9
    move = 0.2
    # reshape to perform vector operations
    xnew = np.zeros(nelx * nely)
    while (l2 - l1) / (l1 + l2) > 1e-3:
        lmid = 0.5 * (l2 + l1)
        xnew[:] = np.maximum(
            0.0,
            np.maximum(
                x - move,
                np.minimum(1.0, np.minimum(x + move, x * np.sqrt(-dc / dv / lmid))),
            ),
        )
        gt = g + np.sum((dv * (xnew - x)))
        if gt > 0:
            l1 = lmid
        else:
            l2 = lmid
    return (xnew, gt)
