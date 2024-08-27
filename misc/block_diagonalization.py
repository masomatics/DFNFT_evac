import torch
import numpy as np
from einops import repeat
import torch
import torch.nn as nn
from einops import repeat
from misc.laplacian import tracenorm_of_normalized_laplacian, make_identity_like
from tqdm import tqdm
import numpy as np

from scipy.sparse.linalg import eigs
from scipy.sparse import csc_matrix


"""
Helper functions for block diagonal analysis
"""


def make_identity(N, D, device):
    if N is None:
        return torch.Tensor(np.array(np.eye(D))).to(device)
    else:
        return torch.Tensor(np.array([np.eye(D)] * N)).to(device)


def make_identity_like(A):
    assert A.shape[-2] == A.shape[-1]  # Ensure A is a batch of squared matrices
    device = A.device
    shape = A.shape[:-2]
    eye = torch.eye(A.shape[-1], device=device)[(None,) * len(shape)]
    return eye.repeat(*shape, 1, 1)


def make_diagonal(vecs):
    vecs = vecs[..., None].repeat(
        *(
            [
                1,
            ]
            * len(vecs.shape)
        ),
        vecs.shape[-1],
    )
    return vecs * make_identity_like(vecs)


# Calculate Normalized Laplacian
def tracenorm_of_normalized_laplacian(A):
    D_vec = torch.sum(A, axis=-1)
    D = make_diagonal(D_vec)
    L = D - A
    inv_A_diag = make_diagonal(1 / torch.sqrt(1e-10 + D_vec))
    L = torch.matmul(inv_A_diag, torch.matmul(L, inv_A_diag))
    sigmas = torch.linalg.svdvals(L)
    return torch.sum(sigmas, axis=-1)


"""
Automated Block Diagonalization (Maehara et al.)
This code is the python rendition of the algorithm publicized at
http://www.misojiro.t.u-tokyo.ac.jp/~maehara/commdec/

"""


def comm(A, X):
    return A @ X - X @ A


def multiply(A, v):
    n = A[0].shape[1]
    N = len(A)
    X = np.reshape(v, (n, n))
    W = np.zeros((n, n))
    for k in range(N):
        W += comm(A[k].T, comm(A[k], X)) + comm(A[k], comm(A[k].T, X))
    W += np.eye(n) * np.trace(X)
    return W.reshape(n * n)


"""
Simultaneously Block Diagonalizing N matrices of shape n x n.
A :the matrix of shape  N x n x n. 
"""


def commdec(A, printSW=1, pickup=10, krylovth=1e-8):
    N = len(A)
    n = A[0].shape[1]

    # Settings
    krylovth = krylovth
    krylovit = n**2
    maxdim = n
    pickup = pickup

    v = [np.random.randn(n**2)]
    v[0] /= np.sqrt(v[0] @ v[0])
    H = csc_matrix((n**2, n**2), dtype=float)

    for j in tqdm(range(krylovit)):
        w = multiply(A, v[j])
        for i in range(max(0, j - 1), j + 1):
            H[i, j] = v[i] @ w
            w -= H[i, j] * v[i]
        a = np.sqrt(w @ w)

        if (a < krylovth) or (j == krylovit - 1):
            break

        H[j + 1, j] = a
        v.append(w / a)

    H = H[: j + 1, : j + 1]
    H = (H + H.T) / 2
    Q = np.column_stack(v)

    d, Y = eigs(H, k=maxdim, which="SM")
    # print(Y.shape, d.shape)

    Y = Y.reshape([Y.shape[0], -1])
    d = np.sqrt(np.diag(d) / (4 * n))
    e = d[pickup - 1]

    X = np.zeros([n**2, 1]).astype(np.complex128)

    for i in range(pickup):
        # print(X.shape, Y[:, [0]].shape)
        X = X + Y[:, [i]]
    X = np.reshape(Q @ X, (n, n))
    D, P = np.linalg.eig(X + X.T)

    if printSW > 0:
        print(f"""err = {e}""")
    if printSW > 1:
        print("small eigs (normalized):")
        len_d = min(3 + pickup, d.shape[0])
        for i in range(len_d):
            print(f"{i+1}: err = {d[i]:.3e}")

    return P, e
