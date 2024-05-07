import numpy as np
import time
"""
    ADMM solver function file. This file including ADMM/OMP/ISTA solver functions which is used for sparse imaging
    author: Da Li
    time: 2024年5月7日13:39:12
"""
def LASSO_admm_solver(x0, A, b, mu, opts=None):
    if opts is None:
        opts = {'maxit': 5000, 'sigma': 0.01, 'ftol': 1e-8, 'gtol': 1e-14, 'gamma': 1.618, 'verbose': 1}
    else:
        default_opts = {'maxit': 5000, 'sigma': 0.01, 'ftol': 1e-8, 'gtol': 1e-14, 'gamma': 1.618, 'verbose': 1}
        for key, value in default_opts.items():
            if key not in opts:
                opts[key] = value

    k = 0
    tt = time.time()
    x = x0.copy()
    out = {}

    m, n = A.shape
    sm = opts['sigma']
    y = np.zeros([n, 256])
    z = np.zeros([n, 256])

    fp = np.inf
    nrmC = np.inf
    f = Func(A, b, mu, x)
    f0 = f
    out['fvec'] = [f0]

    AtA = np.dot(A.conj().T, A)
    R = np.linalg.cholesky(AtA + opts['sigma'] * np.eye(n))
    Atb = np.dot(A.conj().T, b)

    while k < opts['maxit'] and abs(f - fp).all() > opts['ftol'] and nrmC > opts['gtol']:
        fp = f

        w = Atb + sm * z - y
        x = np.linalg.solve(R.conj().T, np.linalg.solve(R, w))

        c = x + y / sm
        z = prox(c, mu / sm)

        y = y + opts['gamma'] * sm * (x - z)
        f = Func(A, b, mu, x)
        nrmC = np.linalg.norm(x - z, 2)

        if opts['verbose']:
            print(f'itr: {k}\tfval: {f}\tfeasi: {nrmC}')

        k += 1
        out['fvec'].append(f)

    out['y'] = y
    out['fval'] = f
    out['itr'] = k
    out['tt'] = time.time() - tt
    out['nrmC'] = np.linalg.norm(c - y, np.inf)

    return x, out

def prox(x, mu):
    y = np.maximum(np.abs(x) - mu, 0)
    return np.sign(x) * y

def Func(A, b, mu, x):
    w = np.dot(A, x) - b
    return 0.5 * np.power(np.linalg.norm(w, 2), 2) + mu * np.linalg.norm(x, 1)