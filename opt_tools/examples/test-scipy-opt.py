import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.optimize as opt

sys.path.append('../..')
import opt_tools as ot


def f(x):
    time.sleep(0.1)
    return [opt.rosen(x), opt.rosen_der(x)]


optlog = ot.OptimisationHelper(
    f,
    [
        ot.tasks.DisplayOptimisation(ot.seq_exp_lin(1.0, 1.0)),
        ot.tasks.LogOptimisation(ot.seq_exp_lin(1.0, 1.0), store_fullg=True, store_x=True),
        ot.tasks.StoreOptimisationHistory('./opthist.pkl', ot.seq_exp_lin(1.0, np.inf, 5.0, 5.0), verbose=True)
    ]
)

x0 = np.array([-5, -5])
optlog.callback(x0)
try:
    xfin = opt.minimize(f, jac=True, x0=x0, method='CG', callback=optlog.callback, options={'gtol': 0.0})
    xfin = opt.minimize(f, jac=True, x0=xfin.x, method='CG', callback=optlog.callback)
    xfin = opt.minimize(f, jac=True, x0=xfin.x, method='CG', callback=optlog.callback)
    finx = xfin.x
    optlog.finish(finx)
    print(xfin)
except KeyboardInterrupt:
    finx = optlog.hist.iloc[-1, :].x
    print(finx)

hist = pd.read_pickle('./opthist.pkl')

plt.figure()
plt.plot(hist.i, hist.f, '-x')
plt.plot(hist.i, hist.gnorm, '-x')
plt.xscale('log')
plt.yscale('log')

plt.figure()
X, Y = np.meshgrid(np.linspace(-5, 5, 500),
                   np.linspace(-5, 5, 500))
xy = np.hstack((X.flatten()[:, None], Y.flatten()[:, None]))
feval = np.array([opt.rosen(v) for v in xy]).reshape(len(X), len(Y))
plt.contour(X, Y, np.log(feval))
hist_points = np.vstack(hist.x)
plt.plot(hist_points[:, 0], hist_points[:, 1], 'x-')

plt.show()
