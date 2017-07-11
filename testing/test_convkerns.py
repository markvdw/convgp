import unittest

import numpy as np

import GPflow
import exp_tools
import svconvgp.convkernels as ckernels


class TestRBFOverride(unittest.TestCase):
    def setUp(self):
        self.k1 = ckernels.Conv(GPflow.kernels.RBF(25), [28, 28], [5, 5])
        self.k2 = ckernels.ConvRBF([28, 28], [5, 5])

        w = np.random.randn(24 * 24)
        self.wk1 = ckernels.WeightedConv(GPflow.kernels.RBF(25), [28, 28], [5, 5])
        self.wk2 = ckernels.WeightedConvRBF([28, 28], [5, 5])
        self.wk1.W = w
        self.wk2.W = w

        self.Z = np.random.randn(5, 25)
        self.X = np.random.randn(10, 28 * 28)

        # set lengthscales and variacnes to non unit values
        self.k1.basekern.variance = 0.001
        self.k2.basekern.variance = 0.001
        self.k1.basekern.lengthscales = 2.34
        self.k2.basekern.lengthscales = 2.34

    def test_Kzx(self):
        result1 = self.k1.compute_Kzx(self.Z, self.X)
        result2 = self.k2.compute_Kzx(self.Z, self.X)
        self.assertTrue(np.allclose(result1, result2))

        result1 = self.wk1.compute_Kzx(self.Z, self.X)
        result2 = self.wk2.compute_Kzx(self.Z, self.X)
        self.assertTrue(np.allclose(result1, result2))

    def test_Kzz(self):
        result1 = self.k1.compute_Kzz(self.Z)
        result2 = self.k2.compute_Kzz(self.Z)
        self.assertTrue(np.allclose(result1, result2))

        result1 = self.wk1.compute_Kzz(self.Z)
        result2 = self.wk2.compute_Kzz(self.Z)
        self.assertTrue(np.allclose(result1, result2))


class TestConv(unittest.TestCase):
    def setUp(self):
        self.mnist = exp_tools.MnistExperiment("abc")
        self.mnist.setup_dataset()
        self.cifar = exp_tools.CifarExperiment("def")
        self.cifar.setup_dataset()

        self.mk = ckernels.Conv(GPflow.kernels.RBF(25), [28, 28], [5, 5])
        self.ck = ckernels.Conv(GPflow.kernels.RBF(25), [32, 32], [5, 5], colour_channels=3)

    def test_num_patches(self):
        for k, e in [(self.mk, self.mnist), (self.ck, self.cifar)]:
            p = k.compute_patches(e.X[:100, :])
            self.assertTrue(p.shape[1] == k.num_patches)


class TestValues(unittest.TestCase):
    """
    TestVariances
    Test that the variances of stationary kernels, for constant inputs are 1.
    """

    def setUp(self):
        self.ks = [ckernels.Conv(GPflow.kernels.RBF(4), [3, 3], [2, 2]),
                   ckernels.WeightedConv(GPflow.kernels.RBF(4), [3, 3], [2, 2]),
                   ckernels.ConvRBF([3, 3], [2, 2])]

    def test_variances(self):
        X = np.ones([10, 3 * 3])
        for k in self.ks:
            self.assertTrue(np.all(k.compute_Kdiag(X) == 1))

    def test_diagonals(self):
        X = np.random.randn(3, 3 * 3)
        for k in self.ks:
            self.assertTrue(np.allclose(k.compute_Kdiag(X), np.diag(k.compute_K_symm(X))))

    def test_weight_vs_conv(self):
        Z = np.random.randn(1000, 4)
        self.assertTrue(np.all(self.ks[0].compute_Kzz(Z) == self.ks[1].compute_Kzz(Z)))


if __name__ == "__main__":
    unittest.main()
