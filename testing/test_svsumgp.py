import unittest

import numpy as np
import tensorflow as tf

import GPflow
import convgp.svsumgp as sumgp

GPflow.settings.numerics.float_type = tf.float64


class TestEquivalence(unittest.TestCase):
    def test_mf_full_equivalence(self):
        # Setup dataset
        X = np.random.rand(100, 1) * 6
        Y = 0.6 * X + 0.3 * np.sin(1.5 * np.pi * X) + np.random.randn(*X.shape) * 0.1
        Y -= np.mean(Y)

        # Initial model to get hyperparameters right
        rbf = GPflow.sgpr.SGPR(X, Y, GPflow.kernels.RBF(1), X[:20, :].copy())
        rbf.kern.lengthscales = 0.5
        rbf.likelihood.variance = 0.1
        rbf.optimize()
        lml = -rbf._objective(rbf.get_free_state())[0]

        mf = sumgp.MeanFieldSVSumGP(X, Y, GPflow.kernels.RBF(1), GPflow.kernels.RBF(1), GPflow.likelihoods.Gaussian(),
                                    rbf.Z.value.copy(), rbf.Z.value.copy())
        mf.kern1.lengthscales = rbf.kern.lengthscales.value.copy()
        mf.kern2.lengthscales = rbf.kern.lengthscales.value.copy()
        mf.kern1.variance = rbf.kern.variance.value.copy() * 0.5
        mf.kern2.variance = rbf.kern.variance.value.copy() * 0.5
        mf.kern1.fixed = True
        mf.kern2.fixed = True
        mf.Z1.fixed = True
        mf.Z2.fixed = True
        mf.optimize()
        mf_lml = -mf._objective(mf.get_free_state())[0]

        full = sumgp.FullSVSumGP(X, Y, GPflow.kernels.RBF(1), GPflow.kernels.RBF(1), GPflow.likelihoods.Gaussian(),
                                 rbf.Z.value.copy(), rbf.Z.value.copy())
        full.kern1.lengthscales = rbf.kern.lengthscales.value.copy()
        full.kern2.lengthscales = rbf.kern.lengthscales.value.copy()
        full.kern1.variance = rbf.kern.variance.value.copy() * 0.5
        full.kern2.variance = rbf.kern.variance.value.copy() * 0.5
        full.likelihood.variance = mf.likelihood.variance.value

        full.q_mu = mf.q_mu.value.T.flatten()[:, None]
        mf_cov = np.zeros((2 * len(mf.Z1.value), 2 * len(mf.Z1.value)))
        mf_cov[:20, :20] = mf.q_sqrt.value[:, :, 0]
        mf_cov[20:, 20:] = mf.q_sqrt.value[:, :, 1]
        full.q_sqrt = mf_cov[:, :, None]

        full.kern1.fixed = True
        full.kern2.fixed = True
        full.Z1.fixed = True
        full.Z2.fixed = True
        full._compile()
        full_lml = -full._objective(full.get_free_state())[0]

        self.assertAlmostEqual(mf_lml, full_lml, 3)

        full.optimize()
        full_opt_lml = -full._objective(full.get_free_state())[0]
        self.assertTrue((lml - full_opt_lml) / full_opt_lml * 100 < 0.2)
