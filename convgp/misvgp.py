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
import convgp.convkernels as ckern
from GPflow import settings

float_type = GPflow.settings.dtypes.float_type


class WeightedMultiChannelConvGP(ckern.Conv):
    def __init__(self, basekern, img_size, patch_size, colour_channels=1):
        ckern.Conv.__init__(self, basekern, img_size, patch_size, colour_channels)
        self.basekern.input_dim = np.prod(patch_size)
        self.W = GPflow.param.Param(np.ones((self.colour_channels, self.num_patches // self.colour_channels)))

    def K(self, X, X2=None):
        Xp = self._get_patches(X)
        Xp = tf.reshape(Xp, (tf.shape(X)[0], self.colour_channels, self.num_patches // self.colour_channels,
                             self.patch_len))
        if X2 is not None:
            raise NotImplementedError

        K = 0
        for c in range(self.colour_channels):
            cK = tf.reshape(self.basekern.K(tf.reshape(Xp[:, c, :, :], (-1, self.patch_len))),
                            (tf.shape(X)[0], self.num_patches // self.colour_channels,
                             tf.shape(X)[0], self.num_patches // self.colour_channels))
            K = K + tf.reduce_sum(cK * self.W[None, c, :, None, None] * self.W[None, None, None, c, :], [1, 3])

        # bigK = self.basekern.K(Xp, Xp2)  # N * num_patches x N * num_patches
        # K = tf.reduce_sum(tf.reshape(bigK, (tf.shape(X)[0], self.num_patches, -1, self.num_patches)), [1, 3])
        return K / self.num_patches ** 2.0

    def Kdiag(self, X):
        Xp = tf.reshape(self._get_patches(X),
                        (tf.shape(X)[0], self.colour_channels, self.num_patches // self.colour_channels,
                         self.patch_len))

        def Kdiag_element(patches):
            # print("")
            # print(patches.get_shape())
            # print("")
            return tf.stack([tf.reduce_sum(self.basekern.K(patches[c, :, :]) * self.W[c, :, None] * self.W[c, None, :])
                             for c in range(self.colour_channels)])

        return tf.reshape(tf.map_fn(Kdiag_element, Xp),
                          (tf.shape(X)[0], self.colour_channels)) / self.num_patches ** 2.0

    def Kzx(self, Z, X):
        p = tf.reshape(self._get_patches(X), (-1, self.patch_len))
        bigKzx = tf.reshape(self.basekern.K(Z, p),
                            (tf.shape(Z)[0], tf.shape(X)[0], self.colour_channels,
                             self.num_patches // self.colour_channels))
        # Kzx = tf.reduce_sum(bigKzx * self.W[None, None, :, :], [-1])  # N x M x colour_channels
        Kzx = tf.einsum('mncp,cp->mnc', bigKzx, self.W)
        return Kzx / self.num_patches

    @property
    def inducing_outputs(self):
        """
        :return: Inducing outputs per input.
        """
        return self.colour_channels


def conditional(Xnew, X, Kdiag, Kmn, Kmm, f, full_cov=False, q_sqrt=None, whiten=False):
    """
    Modification of GPflow.conditionals.conditional to allow evaluated kernel matrices to be passed through.

    Given F, representing the GP at the points X, produce the mean and
    (co-)variance of the GP at the points Xnew.

    Additionally, there may be Gaussian uncertainty about F as represented by
    q_sqrt. In this case `f` represents the mean of the distribution and
    q_sqrt the square-root of the covariance.

    Additionally, the GP may have been centered (whitened) so that
        p(v) = N( 0, I)
        f = L v
    thus
        p(f) = N(0, LL^T) = N(0, K).
    In this case 'f' represents the values taken by v.

    The method can either return the diagonals of the covariance matrix for
    each output of the full covariance matrix (full_cov).

    We assume K independent GPs, represented by the columns of f (and the
    last dimension of q_sqrt).

     - Xnew is a data matrix, size N x D
     - X are data points, size M x D
     - kern is a GPflow kernel
     - f is a data matrix, M x K, representing the function values at X, for K functions.
     - q_sqrt (optional) is a matrix of standard-deviations or Cholesky
       matrices, size M x K or M x M x K
     - whiten (optional) is a boolean: whether to whiten the representation
       as described above.

    These functions are now considered deprecated, subsumed into this one:
        gp_predict
        gaussian_gp_predict
        gp_predict_whitened
        gaussian_gp_predict_whitened

    """

    # compute kernel stuff
    num_data = tf.shape(X)[0]  # M
    num_func = tf.shape(f)[1]  # K

    Lm = tf.cholesky(Kmm)

    # Compute the projection matrix A
    A = tf.matrix_triangular_solve(Lm, Kmn, lower=True)

    # compute the covariance due to the conditioning
    if full_cov:
        fvar = None - tf.matmul(A, A, transpose_a=True)
        shape = tf.stack([num_func, 1, 1])
    else:
        fvar = Kdiag - tf.reduce_sum(tf.square(A), 0)
        shape = tf.stack([num_func, 1])
    fvar = tf.tile(tf.expand_dims(fvar, 0), shape)  # K x N x N or K x N

    # another backsubstitution in the unwhitened case
    if not whiten:
        A = tf.matrix_triangular_solve(tf.transpose(Lm), A, lower=False)

    # construct the conditional mean
    fmean = tf.matmul(A, f, transpose_a=True)

    if q_sqrt is not None:
        if q_sqrt.get_shape().ndims == 2:
            LTA = A * tf.expand_dims(tf.transpose(q_sqrt), 2)  # K x M x N
        elif q_sqrt.get_shape().ndims == 3:
            L = tf.matrix_band_part(tf.transpose(q_sqrt, (2, 0, 1)), -1, 0)  # K x M x M
            A_tiled = tf.tile(tf.expand_dims(A, 0), tf.stack([num_func, 1, 1]))
            LTA = tf.matmul(L, A_tiled, transpose_a=True)  # K x M x N
        else:  # pragma: no cover
            raise ValueError("Bad dimension for q_sqrt: %s" %
                             str(q_sqrt.get_shape().ndims))
        if full_cov:
            fvar = fvar + tf.matmul(LTA, LTA, transpose_a=True)  # K x N x N
        else:
            fvar = fvar + tf.reduce_sum(tf.square(LTA), 1)  # K x N
    fvar = tf.transpose(fvar)  # N x K or N x N x K

    return fmean, fvar


class MultiOutputInducingSVGP(GPflow.svgp.SVGP):
    """
    A very (quick) and dirty implementation of multi output inducing variables idea. The build_predict sums the
    components of each inducing output together.
    """

    def __init__(self, X, Y, kern, likelihood, Z, mean_function=GPflow.mean_functions.Zero(), num_latent=None,
                 q_diag=False, whiten=True, minibatch_size=None):
        super(MultiOutputInducingSVGP, self).__init__(X, Y, kern, likelihood, Z, mean_function, num_latent, q_diag,
                                                      whiten, minibatch_size)

        self.q_mu = GPflow.param.Param(np.zeros((self.num_inducing, self.num_latent * kern.inducing_outputs)))
        q_sqrt = np.array(
            [np.eye(self.num_inducing) for _ in range(self.num_latent * kern.inducing_outputs)]
        ).swapaxes(0, 2).reshape(self.num_inducing, self.num_inducing, kern.inducing_outputs * self.num_latent)
        self.q_sqrt = GPflow.param.Param(q_sqrt,
                                         GPflow.transforms.LowerTriangular(self.num_inducing,
                                                                           kern.inducing_outputs * self.num_latent))

    def build_predict(self, Xnew, full_cov=False):
        mus = []
        vars = []
        q_mu = tf.reshape(self.q_mu, [self.num_inducing, self.num_latent, self.kern.inducing_outputs])
        q_sqrt = tf.reshape(self.q_sqrt,
                            [self.num_inducing, self.num_inducing, self.num_latent, self.kern.inducing_outputs])
        for i in range(self.kern.inducing_outputs):
            Kdiag = self.kern.Kdiag(Xnew)
            Kmn = self.kern.Kzx(self.Z, Xnew)
            Kmm = self.kern.Kzz(self.Z) + tf.eye(tf.shape(self.Z)[0], dtype=float_type) * settings.numerics.jitter_level
            # This next line contains an INCREDIBLY dirty trick. Since Kdiag is only additively used in conditional, we
            # can devide it by the number of components, rather than find the marginal variance contribution for each
            # component separately.
            fmu, fvar = conditional(Xnew, self.Z, Kdiag[:, i], Kmn[:, :, i], Kmm, q_mu[:, :, i],
                                    q_sqrt=q_sqrt[:, :, :, i], full_cov=full_cov, whiten=self.whiten)
            mus.append(fmu)
            vars.append(fvar)

        # mu = tf.reduce_sum(tf.stack(mus), 0)
        # var = tf.reduce_sum(tf.stack(vars), 0)
        mu = tf.add_n(mus)
        var = tf.add_n(vars)
        var = tf.Print(var, [var])
        return mu, var
