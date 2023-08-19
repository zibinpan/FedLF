import torch
import numpy as np
import cvxopt
from cvxopt import matrix
from scipy.optimize import minimize
import time
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def cvxopt_solve_qp(P, q, G=None, h=None, A=None, b=None):

    P = P.astype(np.double)
    q = q.astype(np.double)
    args = [matrix(P), matrix(q)]
    if G is not None:
        args.extend([matrix(G), matrix(h)])
        if A is not None:
            args.extend([matrix(A), matrix(b)])
    sol = cvxopt.solvers.qp(*args)
    optimal_flag = 1
    if 'optimal' not in sol['status']:
        optimal_flag = 0
    return np.array(sol['x']).reshape((P.shape[1],)), optimal_flag


def setup_qp_and_solve(vec, device):

    P = vec @ (vec.T)
    P = P.cpu().detach().numpy()

    n = P.shape[0]
    q = np.zeros(n)

    G = - np.eye(n)
    h = np.zeros(n)

    A = np.ones((1, n))
    b = np.ones(1)

    cvxopt.solvers.options['show_progress'] = False

    sol, optimal_flag = cvxopt_solve_qp(P, q, G, h, A, b)
    sol = torch.from_numpy(sol).float().to(device)
    return sol, optimal_flag


def setup_qp_and_solve_for_mgdaplus(vec, epsilon, lambda0):

    P = np.dot(vec, vec.T)

    n = P.shape[0]
    q = np.zeros(n)

    G = np.vstack([-np.eye(n), np.eye(n)])
    lb = np.array([max(0, lambda0[i] - epsilon) for i in range(n)])
    ub = np.array([min(1, lambda0[i] + epsilon) for i in range(n)])
    h = np.hstack([lb, ub])

    A = np.ones((1, n))
    b = np.ones(1)

    cvxopt.solvers.options['show_progress'] = False
    sol, optimal_flag = cvxopt_solve_qp(P, q, G, h, A, b)
    return sol, optimal_flag


def quadprog(P, q, G, h, A, b):
    P = cvxopt.matrix(P.tolist())
    q = cvxopt.matrix(q.tolist(), tc='d')
    G = cvxopt.matrix(G.tolist())
    h = cvxopt.matrix(h.tolist())
    A = cvxopt.matrix(A.tolist())
    b = cvxopt.matrix(b.tolist(), tc='d')
    cvxopt.solvers.options['show_progress'] = False
    sol = cvxopt.solvers.qp(P, q.T, G.T, h.T, A.T, b)
    return np.array(sol['x'])


def setup_qp_and_solve_for_mgdaplus_1(vec, epsilon, lambda0):

    P = np.dot(vec, vec.T)

    n = P.shape[0]

    q = np.array([[0] for i in range(n)])

    A = np.ones(n).T
    b = np.array([1])

    lb = np.array([max(0, lambda0[i] - epsilon) for i in range(n)])
    ub = np.array([min(1, lambda0[i] + epsilon) for i in range(n)])
    G = np.zeros((2 * n, n))
    for i in range(n):
        G[i][i] = -1
        G[n + i][i] = 1
    h = np.zeros((2 * n, 1))
    for i in range(n):
        h[i] = -lb[i]
        h[n + i] = ub[i]
    sol = quadprog(P, q, G, h, A, b).reshape(-1)

    return sol, 1


def get_d_moomtl_d(grads, device):

    vec = grads
    sol, optimal_flag = setup_qp_and_solve(vec, device)

    d = torch.matmul(sol, grads)

    descent_flag = 1
    c = - (grads @ d)

    if not torch.all(c <= 1e-6):
        descent_flag = 0

    return d, optimal_flag, descent_flag


def get_d_mgdaplus_d(grads, device, epsilon, lambda0):

    vec = grads
    sol, optimal_flag = setup_qp_and_solve_for_mgdaplus_1(
        vec.cpu().detach().numpy(), epsilon, lambda0)

    sol = torch.from_numpy(sol).float().to(device)
    d = sol @ grads

    descent_flag = 1
    c = -(grads @ d)
    if not torch.all(c <= 1e-6):
        descent_flag = 0

    return d, sol, descent_flag


def check_constraints(value, ref_vec, prefer_vec):

    w = ref_vec - prefer_vec

    gx = torch.matmul(w, value/torch.norm(value))
    idx = gx > 0
    return torch.sum(idx), idx


def project(a, b):
    return a @ b / torch.norm(b)**2 * b


def solve_d(Q, g, value, device):
    L = value.cpu().detach().numpy()
    QTg = Q @ g.T
    QTg = QTg.cpu().detach().numpy()
    gTg = g @ g.T
    gTg = gTg.cpu().detach().numpy()

    def fun(x):
        return np.sum((gTg @ x - L)**2)

    cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
            {'type': 'ineq', 'fun': lambda x: x - 1e-10},
            {'type': 'ineq', 'fun': lambda x: QTg @ x}
            )

    x0 = np.random.rand(g.shape[0])
    x0 = x0 / np.sum(x0)
    res = minimize(fun, x0, method='SLSQP', constraints=cons)
    lam = res.x
    lam = torch.from_numpy(lam).float().to(device)
    d = lam @ g
    return d
