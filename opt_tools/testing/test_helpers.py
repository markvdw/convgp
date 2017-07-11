import sys
import time
import unittest

import numpy as np
import numpy.random as rnd

import GPflow

sys.path.append('..')
import opt_tools as ot


class TestGPflowHelper(unittest.TestCase):
    def setUp(self):
        X = np.linspace(0, 5, 100)[:, None]
        Y = 0.3 * np.sin(2 * X) + 0.05 * rnd.randn(*X.shape)

        # model = GPflow.sgpr.SGPR(X, Y, GPflow.kernels.RBF(1), X[rnd.permutation(len(X))[:3], :])
        model = GPflow.sgpr.SGPR(X, Y, GPflow.kernels.RBF(1), X[:3, :].copy())
        model._compile()

        self.optlog = ot.GPflowOptimisationHelper(
            model,
            [
                ot.tasks.DisplayOptimisation(ot.seq_exp_lin(1.0, 1.0)),
                ot.tasks.GPflowLogOptimisation(ot.seq_exp_lin(1.0, 1.0), store_fullg=True, store_x=True),
                ot.tasks.StoreOptimisationHistory('./opthist.pkl', ot.seq_exp_lin(1.0, np.inf, 5.0, 5.0), verbose=True)
            ]
        )

    def test_memoisation(self):
        x = self.optlog.model.get_free_state().copy()

        st = time.time()
        f1, g1 = self.optlog._fg(x)
        t1 = time.time() - st

        st = time.time()
        f2, g2 = self.optlog._fg(x.copy())
        t2 = time.time() - st

        self.assertTrue(t2 < 1e-4)  # Memoised eval should be very quick
        self.assertTrue(t2 / t1 < 1e-2)  # Memoised eval should be more thean 100 times faster
        self.assertTrue(f1 == f2)
        self.assertTrue(np.all(g1 == g2))

        f3, g3 = self.optlog._fg(x.copy() + 0.1)
        self.assertTrue(t2 != f3)
        self.assertTrue(np.all(g2 != g3))
