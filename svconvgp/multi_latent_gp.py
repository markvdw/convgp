# Copyright 2016 James Hensman, Valentine Svensson, alexggmatthews, Mark van der Wilk
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# modified from GPflow/svgp.py

import tensorflow as tf
import numpy as np
from GPflow.model import GPModel
from GPflow.param import Param, ParamList
from GPflow import transforms, conditionals, kullback_leiblers
from GPflow.mean_functions import Zero
from GPflow.tf_wraps import eye
from GPflow._settings import settings
from GPflow.minibatch import MinibatchData


class SVGP(GPModel):
    """
    This is the very similar to GPflow.svgp, but in this case the different
    column of Y are modelled with different kernels.

    The inducing/pseudo-point Z are shared between processes.
    """
    def __init__(self, X, Y, kerns, likelihood, Z, mean_function=Zero(),
                 q_diag=False, whiten=True, minibatch_size=None):
        """
        - X is a data matrix, size N x D
        - Y is a data matrix, size N x R
        - kerns is a list of GPflow kernels
        - likelihood, mean_function are appropriate GPflow objects
        - Z is a matrix of pseudo inputs, size M x D
        - whiten is a boolean. If True, we use the whitened representation of
          the inducing points.
        """
        # sort out the X, Y into MiniBatch objects.
        if minibatch_size is None:
            minibatch_size = X.shape[0]
        self.num_data = X.shape[0]
        X = MinibatchData(X, minibatch_size, np.random.RandomState(0))
        Y = MinibatchData(Y, minibatch_size, np.random.RandomState(0))

        # init the super class, accept args
        GPModel.__init__(self, X, Y, None, likelihood, mean_function)
        self.kerns = ParamList(kerns)
        self.q_diag, self.whiten = q_diag, whiten
        self.Z = Param(Z)
        self.num_latent = len(kerns)
        self.num_inducing = Z.shape[0]

        # init variational parameters
        self.q_mu = Param(np.zeros((self.num_inducing, self.num_latent)))
        if self.q_diag:
            self.q_sqrt = Param(np.ones((self.num_inducing, self.num_latent)),
                                transforms.positive)
        else:
            q_sqrt = np.array([np.eye(self.num_inducing)
                               for _ in range(self.num_latent)]).swapaxes(0, 2)
            self.q_sqrt = Param(q_sqrt)  # , transforms.LowerTriangular(q_sqrt.shape[2]))  # Temp remove transform

    def build_prior_KL(self):
        if self.whiten:
            if self.q_diag:
                KL = kullback_leiblers.gauss_kl_white_diag(self.q_mu, self.q_sqrt)
            else:
                KL = kullback_leiblers.gauss_kl_white(self.q_mu, self.q_sqrt)
        else:
            KL = 0.
            for i, k in enumerate(self.kerns):
                K = k.K(self.Z) + eye(self.num_inducing) * settings.numerics.jitter_level
                if self.q_diag:
                    KL += kullback_leiblers.gauss_kl_diag(self.q_mu[:, i:i+1], self.q_sqrt[:, i:i+1], K)
                else:
                    KL += kullback_leiblers.gauss_kl(self.q_mu[:, i:i+1], self.q_sqrt[:, i:i+1], K)
        return KL

    def build_likelihood(self):
        """
        This gives a variational bound on the model likelihood.
        """

        # Get prior KL.
        KL = self.build_prior_KL()

        # Get conditionals
        fmean, fvar = self.build_predict(self.X, full_cov=False)

        # Get variational expectations.
        var_exp = self.likelihood.variational_expectations(fmean, fvar, self.Y)

        # re-scale for minibatch size
        scale = tf.cast(self.num_data, settings.dtypes.float_type) /\
            tf.cast(tf.shape(self.X)[0], settings.dtypes.float_type)

        return tf.reduce_sum(var_exp) * scale - KL

    def build_predict(self, Xnew, full_cov=False):
        mu, var = [], []
        for i, k in enumerate(self.kerns):
            q_sqrt = self.q_sqrt[:, :,  i:i+1]  # will fail if self.qdiag
            mu_i, var_i = conditionals.conditional(Xnew, self.Z, k, self.q_mu[:, i:i+1],
                                                   q_sqrt=q_sqrt, full_cov=full_cov, whiten=self.whiten)
            mu.append(mu_i)
            var.append(var_i)
        return tf.concat_v2(mu, 1) + self.mean_function(Xnew), tf.concat_v2(var, 1)

if __name__ == "__main__":
    import GPflow
    from matplotlib import pyplot as plt

    X = np.random.rand(1000, 1)
    Y1 = np.sin(2 * np.pi * 4 * X)
    Y2 = np.cos(2 * np.pi * X)
    Y = np.hstack((Y1, Y2))
    Z = X[::10]

    kerns = [GPflow.kernels.Matern32(1), GPflow.kernels.Matern12(1)]
    lik = GPflow.likelihoods.Gaussian()
    lik.variance = 0.001

    m = SVGP(X, Y, kerns, lik, Z)

    m.optimize(maxiter=200)

    Xtest = np.linspace(-0.1, 1.1, 100).reshape(-1, 1)
    mu, var = m.predict_y(Xtest)
    plt.plot(Xtest, mu[:, 0], 'b', lw=2)
    plt.plot(Xtest, mu[:, 0] + 2*np.sqrt(var[:, 0]), 'b--', lw=1)
    plt.plot(Xtest, mu[:, 0] - 2*np.sqrt(var[:, 0]), 'b--', lw=1)
    plt.plot(X, Y[:, 0], 'kx', mew=2)
    plt.plot(Xtest, mu[:, 1], 'b', lw=2)
    plt.plot(Xtest, mu[:, 1] + 2*np.sqrt(var[:, 1]), 'b--', lw=1)
    plt.plot(Xtest, mu[:, 1] - 2*np.sqrt(var[:, 1]), 'b--', lw=1)
    plt.plot(X, Y[:, 1], 'kx', mew=2)
