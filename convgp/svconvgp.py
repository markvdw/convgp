import numpy as np
import tensorflow as tf

import GPflow
from GPflow.param import Param

float_type = GPflow.settings.dtypes.float_type


class SVColourConvGP(GPflow.svgp.SVGP):
    """
    SVColourConvGP
    Should be the same as an additive model over the colour cube.
    """

    def __init__(self, X, Y, kern, likelihood, Z, colour_channels, mean_function=GPflow.mean_functions.Zero(),
                 num_latent=None, q_diag=False, whiten=True, minibatch_size=None):
        assert not q_diag
        super(SVColourConvGP, self).__init__(X, Y, kern, likelihood, Z, mean_function, num_latent, q_diag, whiten,
                                             minibatch_size)

        # init variational parameters
        self.q_mu = Param(np.zeros((self.num_inducing, self.num_latent * colour_channels)))
        q_sqrt = np.array(
            [np.eye(self.num_inducing) for _ in range(self.num_latent * colour_channels)]
        ).swapaxes(0, 2).reshape(self.num_inducing, self.num_inducing, self.num_latent * colour_channels)
        self.q_sqrt = Param(q_sqrt)  # , transforms.LowerTriangular(q_sqrt.shape[2]))  # Temp remove transform

        self.colour_channels = colour_channels
        self.wc = GPflow.param.Param(np.ones(colour_channels))

    def build_predict(self, Xnew, full_cov=False):
        assert type(self.mean_function) is GPflow.mean_functions.Zero
        Xnew = tf.reshape(Xnew, [tf.shape(Xnew)[0], -1, self.colour_channels])
        mus = []
        vars = []
        q_mu = tf.reshape(self.q_mu, [self.num_inducing, self.num_latent, self.colour_channels])
        q_sqrt = tf.reshape(self.q_sqrt, [self.num_inducing, self.num_inducing, self.num_latent, self.colour_channels])
        for c in range(self.colour_channels):
            fmu, fvar = GPflow.conditionals.conditional(Xnew[:, :, c], self.Z, self.kern, q_mu[:, :, c],
                                                        q_sqrt=q_sqrt[:, :, :, c], full_cov=full_cov,
                                                        whiten=self.whiten)
            mus.append(fmu)
            vars.append(fvar)

        mu = tf.reduce_sum(tf.stack(mus) * self.wc[:, None, None], 0)
        var = tf.reduce_sum(tf.stack(vars) * self.wc[:, None, None], 0)
        return mu, var


class SVConvGP(GPflow.svgp.SVGP):
    def __init__(self, X, Y, kern, likelihood, Z, patch_size, minibatch_size=None):
        """
        SVConvGP
        :param X: Input images
        :param Y: Output images
        :param kern: Kernel
        :param likelihood: Likelihood
        :param Z: Inducing patches (MxP^2) or int.
        :param patch_size:
        :param minibatch_size:
        """
        self.patch_size = patch_size
        self.minibatch_size = minibatch_size

        assert kern.input_dim == patch_size ** 2.0

        GPflow.svgp.SVGP.__init__(self, X, Y, kern, likelihood, np.zeros((Z if type(Z) is int else Z, kern.input_dim)),
                                  minibatch_size=minibatch_size)
        if type(Z) is int:
            # Find initialised patches here
            self.Z = GPflow.param.Param(self.compute_patches(X[:Z, :])[:, 0, :])
        else:
            self.Z = GPflow.param.Param(Z)

    def build_likelihood(self):
        # Get prior KL.
        KL = self.build_prior_KL()

        # get q(f) for each image in the minibatch:
        mu, var = self.build_predict(self.X)

        # Get variational expectations.
        var_exp = self.likelihood.variational_expectations(mu, var, self.Y)

        # re-scale for minibatch size
        scale = tf.cast(self.num_data, float_type) / tf.cast(tf.shape(self.X)[0], float_type)

        return tf.reduce_sum(var_exp) * scale - KL

    def build_predict(self, Xnew, full_cov=False):

        # we cannot do full_cov (yet).
        # Note that the full_cov arg in the line below is across all patches within an image.
        assert not full_cov

        # fetch the patches for each image in the minibatch
        patches = self._get_patches(Xnew)  # batch_size x num_patches x patch_dims

        # define a function that predicts the mean and variance for one image
        def build_predict_image(Xpatches):
            mu, var = GPflow.conditionals.conditional(Xpatches, self.Z, self.kern, self.q_mu,
                                                      q_sqrt=self.q_sqrt, full_cov=True, whiten=self.whiten)
            return tf.reduce_sum(mu, [0]), tf.reduce_sum(var, [0, 1])

        # loop over images with map_fn
        return tf.map_fn(build_predict_image, patches, dtype=(float_type, float_type))

    def _get_patches(self, X):
        patches = tf.extract_image_patches(tf.reshape(X, [-1, 28, 28, 1]),
                                           [1, self.patch_size, self.patch_size, 1],
                                           [1, 1, 1, 1],
                                           [1, 1, 1, 1], "VALID")
        shp = tf.shape(patches)
        return tf.reshape(patches, [shp[0], shp[1] * shp[2], shp[3]])

    @GPflow.param.AutoFlow((float_type,))
    def compute_patches(self, X):
        return self._get_patches(X)

    @GPflow.param.AutoFlow((float_type, [None, None]))
    def compute_patch_predictions(self, Xnew):
        patches = self._get_patches(Xnew)  # BxPxD

        def bp(x):
            return GPflow.conditionals.conditional(x, self.Z, self.kern, self.q_mu,
                                                   q_sqrt=self.q_sqrt, full_cov=False, whiten=self.whiten)[0]

        return tf.map_fn(bp, patches)
