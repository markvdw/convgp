import numpy as np

import unittest

import tensorflow as tf

import GPflow

import svconvgp.likelihoods as lik

float_type = GPflow.settings.dtypes.float_type
int_type = GPflow.settings.dtypes.int_type


def softmax(Y, F, w):
    logits = np.exp(np.dot(w, F.T).T)
    return np.log((logits / logits.sum(1)[:, None])[np.arange(len(F)), Y[:, 0]])


class TestProjectedLikelihoodVsSelf(unittest.TestCase):
    def setUp(self):
        num_latent = 2
        classes = 3
        self.l = lik.MultiClassProjection(np.random.randn(classes, num_latent))
        self.Fmu = np.random.randn(10, num_latent)
        self.Fvar = 0.0 * np.random.rand(10, num_latent) + 0.00
        self.Y = np.random.randint(0, 3, (self.Fmu.shape[0], 1)) * 0 + 1

        self.af_logp = GPflow.param.AutoFlow((float_type,), (int_type,))(lik.MultiClassProjection.logp)
        self.af_ve = GPflow.param.AutoFlow((float_type,), (float_type,), (int_type,))(
            lik.MultiClassProjection.variational_expectations
        )

    def test_logp(self):
        # fvals = np.array([[1, 1],
        #                   [0, 1],
        #                   [1, 0],
        #                   [0.1, 0.1],
        #                   [99, 99],
        #                   [-99, -99],
        #                   [0, 0]])
        logits = np.exp(np.dot(self.l.weights.value, self.Fmu.T).T)
        nplogp = np.log((logits / logits.sum(1)[:, None])[np.arange(len(self.Fmu)), self.Y[:, 0]])[:, None]
        tflogp = self.af_logp(self.l, self.Fmu, self.Y)
        self.assertTrue(np.allclose(nplogp, tflogp))

    def test_var_exp(self):
        self.assertTrue(np.allclose(
            self.af_ve(self.l, self.Fmu, self.Fvar * 0.0, self.Y),
            self.af_logp(self.l, self.Fmu, self.Y)
        ))  # Monte carlo with zero variance should be the evaluation


class TestProjectedLikelihoodVsBernoulli(unittest.TestCase):
    def setUp(self):
        num_latent = 1
        self.l = lik.MultiClassProjection(np.array([-0.5, 0.5])[:, None])
        self.l.mc_samples = 100000
        self.Fmu = np.random.randn(10, num_latent)
        self.Fvar = 1.0 * np.random.rand(10, num_latent) + 0.1
        self.Y = np.random.randint(0, 3, (self.Fmu.shape[0], 1)) * 0 + 1

        self.af_ve = GPflow.param.AutoFlow((float_type,), (float_type,), (int_type,))(
            lik.MultiClassProjection.variational_expectations
        )

    def test_var_exp(self):
        lik = GPflow.likelihoods.Bernoulli(lambda x: 1.0 / (1 + tf.exp(-x)))
        af_b_ve = GPflow.param.AutoFlow((float_type,), (float_type,), (int_type,))(
            GPflow.likelihoods.Bernoulli.variational_expectations
        )

        a = af_b_ve(lik, self.Fmu, self.Fvar, self.Y)
        b = self.af_ve(self.l, self.Fmu, self.Fvar, self.Y)
        maxpd = np.max(np.abs((a - b) / b)) * 100
        self.assertTrue(maxpd < 1.0)

