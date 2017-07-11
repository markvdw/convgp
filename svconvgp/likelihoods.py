import numpy as np

import tensorflow as tf

import GPflow


class MultiClassProjection(GPflow.likelihoods.Likelihood):
    def __init__(self, weights):
        """
        __init__
        :param weights: Classes x num_latent
        """
        super(MultiClassProjection, self).__init__()
        self.weights = GPflow.param.Param(weights)  # Classes x num_latent
        self.classes = self.weights.shape[0]
        self.mc_samples = 10

    def logp(self, F, Y):
        """
        logp
        logp = softmax(Y; W F)
        :param F: N x num_latent
        :param Y: N x 1
        :return:
        """
        # TODO: Assert that Y is of the correct shape
        logits = tf.matmul(F, self.weights, transpose_b=True)
        return -tf.nn.sparse_softmax_cross_entropy_with_logits(labels=Y[:, 0], logits=logits)[:, None]

    def variational_expectations(self, Fmu, Fvar, Y):
        """
        variational_expectations
        :param Fmu: N x num_latent
        :param Fvar: N x num_latent
        :param Y: N x 1
        :return:
        """
        # Calculate variational expectations using Monte Carlo
        samps = (Fmu[None, :, :] + Fvar[None, :, :] ** 0.5 *
                 tf.random_normal(tf.concat([[self.mc_samples], tf.shape(Fmu)], 0),
                                  dtype=GPflow.settings.dtypes.float_type))
        sampsr = tf.reshape(samps, (self.mc_samples * tf.shape(Fmu)[0], tf.shape(Fmu)[1]))
        samp_logps = tf.reshape(self.logp(sampsr, tf.tile(Y, [self.mc_samples, 1])),
                                [self.mc_samples, tf.shape(Fmu)[0]])
        return tf.reduce_mean(samp_logps, 0)[:, None]

    def predict_mean_and_var(self, Fmu, Fvar):
        samps = (Fmu[None, :, :] + Fvar[None, :, :] ** 0.5 *
                 tf.random_normal(tf.concat([[self.mc_samples], tf.shape(Fmu)], 0),
                                  dtype=GPflow.settings.dtypes.float_type))
        sampsr = tf.reshape(samps, (self.mc_samples * tf.shape(Fmu)[0], tf.shape(Fmu)[1]))
        logits = tf.matmul(sampsr, self.weights, transpose_b=True)  # mc-samples*N x Classes
        p = tf.reduce_sum(tf.reshape(tf.nn.softmax(logits), [self.mc_samples, tf.shape(Fmu)[0], self.classes]))
        return p, p - p ** 2.0

    def predict_density(self, Fmu, Fvar, Y):
        raise NotImplementedError

    def conditional_mean(self, F):
        raise NotImplementedError

    def conditional_variance(self, F):
        raise NotImplementedError
