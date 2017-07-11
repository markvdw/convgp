import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import numpy.random as rnd
import pandas as pd

import GPflow

sys.path.append('..')
import opt_tools as ot

rnd.seed(4)

X = np.linspace(0, 5, 100)[:, None]
Y = 0.3 * np.sin(2 * X) + 0.05 * rnd.randn(*X.shape)

# model = GPflow.sgpr.SGPR(X, Y, GPflow.kernels.RBF(1), X[rnd.permutation(len(X))[:3], :])
model = GPflow.sgpr.SGPR(X, Y, GPflow.kernels.RBF(1), X[:3, :].copy())
model._compile()

optlog = ot.GPflowOptimisationHelper(
    model,
    [
        ot.tasks.DisplayOptimisation(ot.seq_exp_lin(1.0, 1.0)),
        ot.tasks.GPflowLogOptimisation(ot.seq_exp_lin(1.0, 1.0), store_fullg=True, store_x=True),
        ot.tasks.StoreOptimisationHistory('./opthist.pkl', ot.seq_exp_lin(1.0, np.inf, 5.0, 5.0), verbose=True)
    ]
)
optlog.callback(model.get_free_state())

# Start optimisation
model.optimize(callback=optlog.callback, disp=False, maxiter=10)
optlog.finish(model.get_free_state())
print("Finished first optimisation run. %i iterations." % optlog.hist.i.max())
print("")

# Resume optimisation
hist = pd.read_pickle('./opthist.pkl')
time.sleep(3.0)
model._wrapped_objective = model._objective


def obj(x):
    time.sleep(0.2)
    return model._wrapped_objective(x)


model._objective = obj
optlog = ot.GPflowOptimisationHelper(
    model,
    [
        ot.tasks.DisplayOptimisation(ot.seq_exp_lin(1.0, 1.0)),
        ot.tasks.GPflowLogOptimisation(ot.seq_exp_lin(1.0, 1.0), store_fullg=True, store_x=True, hist=hist),
        ot.tasks.StoreOptimisationHistory('./opthist.pkl', ot.seq_exp_lin(1.0, np.inf, 5.0, 5.0), verbose=True),
        ot.tasks.Timeout(5.0)
    ]
)
try:
    model.optimize(callback=optlog.callback, disp=False)
except ot.OptimisationTimeout:
    print("Optimisation timeout...")
except KeyboardInterrupt:
    print("Cancelled by user...")
    optlog.finish(model.get_free_state())

print("Plotting...")
hist = pd.read_pickle('./opthist.pkl')

plt.figure(figsize=(14, 5))
plt.subplot(131)
plt.plot(hist.i, hist.f, '-x')
plt.xscale('log')

plt.subplot(132)
plt.plot(hist.i, hist.gnorm, '-x')
plt.xscale('log')
plt.yscale('log')

plt.subplot(133)
plt.plot(X, Y, 'x')
pY, pYv = model.predict_y(X)
plt.plot(X, pY)
plt.plot(X, pY + pYv ** 0.5 * 2, 'r')
plt.plot(X, pY - pYv ** 0.5 * 2, 'r')

plt.show()
