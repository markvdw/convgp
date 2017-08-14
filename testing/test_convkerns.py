import unittest

import numpy as np

import GPflow
import convgp.convkernels as ckernels


class TestConvRBF(unittest.TestCase):
    def setUp(self):
        k = ckernels.Conv(GPflow.kernels.RBF(25), [28, 28], [5, 5])
        kr = ckernels.ConvRBF([28, 28], [5, 5])

        w = np.random.randn(24 * 24)
        wk = ckernels.WeightedConv(GPflow.kernels.RBF(25), [28, 28], [5, 5])
        wkr = ckernels.WeightedConvRBF([28, 28], [5, 5])
        wk.W = w
        wkr.W = w

        self.testkerns = [kr, wkr]
        self.refkerns = [k, wk]

        self.Z = np.random.randn(5, 25)
        self.X = np.random.randn(10, 28 * 28)

        # set lengthscales and variances to non unit values
        k.basekern.variance = 0.001
        kr.basekern.variance = 0.001
        k.basekern.lengthscales = 2.34
        kr.basekern.lengthscales = 2.34

    def test_Kzx(self):
        for k1, k2 in zip(self.testkerns, self.refkerns):
            self.assertTrue(np.allclose(k1.compute_Kzx(self.Z, self.X), k2.compute_Kzx(self.Z, self.X)))

    def test_Kzz(self):
        for k1, k2 in zip(self.testkerns, self.refkerns):
            self.assertTrue(np.allclose(k1.compute_Kzx(self.Z, self.X), k2.compute_Kzx(self.Z, self.X)))


class TestGeneral(unittest.TestCase):
    def setUp(self):
        wconv = ckernels.WeightedConv(GPflow.kernels.RBF(4), [3, 3], [2, 2])
        mcwconv = ckernels.WeightedMultiChannelConvGP(GPflow.kernels.RBF(4), [3, 3], [2, 2])
        self.kernels = [ckernels.Conv(GPflow.kernels.RBF(4), [3, 3], [2, 2]),
                        wconv,
                        ckernels.ConvRBF([3, 3], [2, 2]),
                        mcwconv]

    def test_diag_consistency(self):
        X = np.random.randn(3, 3 * 3)
        for k in self.kernels:
            self.assertTrue(np.allclose(k.compute_Kdiag(X), np.diag(k.compute_K_symm(X))))
            self.assertTrue(np.all(k.compute_Kdiag(np.ones([10, 3 * 3])) == 1.0))

    def test_init_patches(self):
        X = np.zeros((1, 9))
        X[0, 0] = 1.0
        for k in self.kernels:
            self.assertTrue(k.init_inducing(X, 10, "default").shape == (4, k.patch_len))
            self.assertTrue(k.init_inducing(X, 10, "patches-unique").shape == (2, k.patch_len))


class TestColourChannels(unittest.TestCase):
    def setUp(self):
        self.psz = 2  # patch size
        self.isz = 3  # img size
        self.imgs = 4
        self.k1 = ckernels.Conv(GPflow.kernels.RBF(self.psz * self.psz), [self.isz, self.isz], [self.psz, self.psz],
                                colour_channels=2)
        self.k2 = ckernels.ColourPatchConv(GPflow.kernels.RBF(self.psz * self.psz * 2), [self.isz, self.isz],
                                           [self.psz, self.psz], colour_channels=2)
        base_img = np.arange(self.imgs * self.isz * self.isz).reshape(self.imgs, self.isz * self.isz, 1)
        self.X = np.concatenate((base_img, base_img + 0.25), -1)

    def test_colour_channels(self):
        p1 = self.k1.compute_patches(self.X)
        p2 = self.k2.compute_patches(self.X)
        patches = self.k2.num_patches

        # To get the individual patches from p2, we reshape the last axis to separate out the colour channels. Then, we
        # roll the colour channels next to the image axis, and we end up with the same ordering as Conv. This shows that
        # The last dimension of MultiChannelConv contains combinations of patches (from different colour channels but
        # the same position) that are treated separately by Conv.

        p2r = p2.reshape(self.imgs, patches, self.psz * self.psz, 2)

        self.assertTrue(np.max(np.abs(
            p1 - p2r.transpose([0, 3, 1, 2]).reshape(self.imgs, patches * 2, self.psz * self.psz)
        )) == 0)

    def test_num_patches(self):
        self.assertTrue(self.k1.num_patches == 2 * self.k2.num_patches)

        klist = [self.k1, self.k2]
        patches = [k.compute_patches(self.X) for k in klist]
        num_patches = [p.shape[1] for p in patches]
        patch_len = [p.shape[2] for p in patches]
        self.assertTrue(num_patches == [k.num_patches for k in klist])
        self.assertTrue(patch_len == [k.patch_len for k in klist])


class TestWeightedConv(unittest.TestCase):
    def setUp(self):
        self.ks = [ckernels.Conv(GPflow.kernels.RBF(4), [3, 3], [2, 2], colour_channels=2),
                   ckernels.ColourPatchConv(GPflow.kernels.RBF(8), [3, 3], [2, 2], colour_channels=2)]
        self.wks = [ckernels.WeightedConv(GPflow.kernels.RBF(4), [3, 3], [2, 2], colour_channels=2),
                    ckernels.WeightedColourPatchConv(GPflow.kernels.RBF(8), [3, 3], [2, 2], colour_channels=2)]

    def test_weight_vs_conv(self):
        X = np.random.randn(3, 9, 2).reshape(3, -1)
        for k, wk in zip(self.ks, self.wks):
            assert k.patch_len == wk.patch_len
            Z = np.random.randn(8, k.patch_len)
            self.assertTrue(np.all(k.compute_Kzx(Z, X) == wk.compute_Kzx(Z, X)))
            self.assertTrue(np.all(k.compute_Kzz(Z) == wk.compute_Kzz(Z)))

    def test_weight_inversion(self):
        X = np.random.randn(3, 9, 2).reshape(3, -1)
        for k, wk in zip(self.ks, self.wks):
            assert k.patch_len == wk.patch_len
            Z = np.random.randn(8, k.patch_len)
            wk.W = -1 * np.ones(wk.W.shape)
            self.assertTrue(np.all(k.compute_Kzx(Z, X) == -1 * wk.compute_Kzx(Z, X)))
            self.assertTrue(np.all(k.compute_Kzz(Z) == wk.compute_Kzz(Z)))
            wk.W = np.ones(wk.W.shape)


class TestWeightedConsistency(unittest.TestCase):
    def setUp(self):
        wconv = ckernels.WeightedConv(GPflow.kernels.RBF(4), [5, 5], [2, 2])
        wconv.W = np.random.randn(wconv.num_patches)
        self.kernels = [wconv]

    def test_diag_consistency(self):
        X = np.random.randn(7, 5 * 5)
        for k in self.kernels:
            self.assertTrue(np.allclose(k.compute_Kdiag(X), np.diag(k.compute_K_symm(X))))


if __name__ == "__main__":
    unittest.main()
