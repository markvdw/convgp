# Copyright 2017 Mark van der Wilk, James Hensman
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


import numpy as np
import tensorflow as tf

import GPflow
import GPflow.conditionals
from GPflow import settings

float_type = GPflow.settings.dtypes.float_type


class MeanFieldSVSumGP(GPflow.svgp.SVGP):
    def __init__(self, X, Y, k1, k2, likelihood, Z1, Z2, mean_function=GPflow.mean_functions.Zero(),
                 num_latent=None, q_diag=False, whiten=True, minibatch_size=None):
        super(MeanFieldSVSumGP, self).__init__(X, Y, None, likelihood, Z1.copy(), mean_function, num_latent, q_diag,
                                               whiten, minibatch_size)
        del self.Z
        del self.kern
        self.kern1 = k1  # For now, just hard-code two kernels
        self.kern2 = k2
        self.Z1 = GPflow.param.Param(Z1)
        self.Z2 = GPflow.param.Param(Z2)
        self.q_mu = GPflow.param.Param(np.zeros((self.Z1.shape[0], self.num_latent * 2)))
        q_sqrt = np.array(
            [np.eye(self.num_inducing) for _ in range(self.num_latent * 2)]
        ).swapaxes(0, 2).reshape(self.num_inducing, self.num_inducing, self.num_latent * 2)
        self.q_sqrt = GPflow.param.Param(q_sqrt,
                                         GPflow.transforms.LowerTriangular(self.num_inducing, self.num_latent * 2))

    def build_predict(self, Xnew, full_cov=False):
        mus = []
        vars = []
        q_mu = tf.reshape(self.q_mu, [self.num_inducing, self.num_latent, 2])
        q_sqrt = tf.reshape(self.q_sqrt,
                            [self.num_inducing, self.num_inducing, self.num_latent, 2])
        for i, (k, Z) in enumerate(zip([self.kern1, self.kern2], [self.Z1, self.Z2])):
            fmu, fvar = GPflow.conditionals.conditional(Xnew, Z, k, q_mu[:, :, i], q_sqrt=q_sqrt[:, :, :, i],
                                                        full_cov=full_cov, whiten=self.whiten)
            mus.append(fmu)
            vars.append(fvar)

        mu = tf.reduce_sum(tf.stack(mus), 0)
        var = tf.reduce_sum(tf.stack(vars), 0)
        return mu, var


class FullSVSumGP(MeanFieldSVSumGP):
    def __init__(self, X, Y, k1, k2, likelihood, Z1, Z2, mean_function=GPflow.mean_functions.Zero(),
                 num_latent=None, q_diag=False, whiten=True, minibatch_size=None):
        super(FullSVSumGP, self).__init__(X, Y, k1, k2, likelihood, Z1, Z2, mean_function, num_latent, q_diag, whiten,
                                          minibatch_size)
        del self.num_inducing
        inducing_variables = self.Z1.shape[0] + self.Z2.shape[0]
        self.q_mu = GPflow.param.Param(np.zeros((inducing_variables, self.num_latent)))
        q_sqrt = np.array(
            [np.eye(inducing_variables) for _ in range(self.num_latent)]
        ).swapaxes(0, 2).reshape(inducing_variables, inducing_variables, self.num_latent)
        self.q_sqrt = GPflow.param.Param(q_sqrt, GPflow.transforms.LowerTriangular(inducing_variables, self.num_latent))

    def build_predict(self, Xnew, full_cov=False):
        Kmn1 = self.kern1.Kzx(self.Z1, Xnew)
        Kmm1 = self.kern1.Kzz(self.Z1) + tf.eye(tf.shape(self.Z1)[0], dtype=float_type) * settings.numerics.jitter_level
        Lm1 = tf.cholesky(Kmm1)

        Kmn2 = self.kern2.Kzx(self.Z2, Xnew)
        Kmm2 = self.kern2.Kzz(self.Z2) + tf.eye(tf.shape(self.Z2)[0], dtype=float_type) * settings.numerics.jitter_level
        Lm2 = tf.cholesky(Kmm2)

        A1 = tf.matrix_triangular_solve(Lm1, Kmn1)
        A2 = tf.matrix_triangular_solve(Lm2, Kmn2)  # MxN
        A = tf.concat([A1, A2], 0)  # 2MxN

        diags = self.kern1.Kdiag(Xnew) + self.kern2.Kdiag(Xnew)
        fvar = diags - tf.reduce_sum(A1 ** 2.0, 0) - tf.reduce_sum(A2 ** 2.0, 0)
        fvar = tf.tile(fvar[None, :], (self.num_latent, 1))
        fmean = tf.matmul(A, self.q_mu, transpose_a=True)

        L = tf.matrix_band_part(tf.transpose(self.q_sqrt, (2, 0, 1)), -1, 0)
        A_tiled = tf.tile(tf.expand_dims(A, 0), (self.num_latent, 1, 1))
        LTA = tf.matmul(L, A_tiled, transpose_a=True)

        fvar = tf.transpose(fvar + tf.reduce_sum(tf.square(LTA), 1))

        return fmean, fvar
