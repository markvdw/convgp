"""
vanilla-mnist.py
Run the convolutional GP on MNIST, using the standard multi-label classification output.
"""
import argparse
import itertools
import os

import numpy as np
import pandas as pd

import GPflow
import exp_tools
import opt_tools
import svconvgp.convkernels as ckern


class Mnist01Experiment(exp_tools.MnistExperiment):
    def __init__(self, name=None, M=100, run_settings=None):
        name = "mnist01-%s%i" % (run_settings['kernel'], M) if name is None else name
        super(Mnist01Experiment, self).__init__(name)
        self.run_settings = run_settings if run_settings is not None else {}
        self.M = M

    def setup_dataset(self, verbose=False):
        super(Mnist01Experiment, self).setup_dataset()

        def filter_01(X, Y):
            lbls01 = np.logical_or(Y == 0, Y == 1).flatten()
            return X[lbls01, :], Y[lbls01]

        self.X, self.Y = filter_01(self.X, self.Y)
        self.Xt, self.Yt = filter_01(self.Xt, self.Yt)

    def setup_model(self):
        Z = None
        if self.run_settings['kernel'] == "rbf":
            k = GPflow.kernels.RBF(28 * 28, ARD=self.run_settings['kernel_ard'])
            Z = self.X[np.random.permutation(len(self.X))[:self.M], :]
            k.lengthscales = 3.0
        elif self.run_settings['kernel'] == "conv":
            # k = ckern.ConvRBF([28, 28], [3, 3]) + GPflow.kernels.White(1, 1e-3)
            k = ckern.Conv(GPflow.kernels.RBF(9, ARD=self.run_settings['kernel_ard']), [28, 28],
                           [3, 3]) + GPflow.kernels.White(1, 1e-3)
        elif self.run_settings['kernel'] == "wconv":
            k = ckern.WeightedConv(GPflow.kernels.RBF(9, ARD=self.run_settings['kernel_ard']), [28, 28],
                                   [3, 3]) + GPflow.kernels.White(1, 1e-3)
        else:
            raise NotImplementedError

        if Z is None:
            if self.run_settings['Zinit'] == "random":
                Z = np.random.rand(self.M, 9)
            elif self.run_settings['Zinit'] == "patches":
                subsetX = self.X[np.random.permutation(len(self.X))[:self.M], :]
                patches = k.kern_list[0].compute_patches(subsetX).reshape(-1, k.kern_list[0].patch_len)
                Z = patches[np.random.permutation(len(patches))[:self.M], :]
            elif self.run_settings['Zinit'] == "patches-unique":
                Z = np.zeros((1, k.kern_list[0].patch_len))
                for x in np.split(self.X, len(self.X) // 100):
                    patches = k.kern_list[0].compute_patches(x).reshape(-1, k.kern_list[0].patch_len)
                    Z = np.vstack({tuple(row) for row in np.vstack((Z, patches))})
                Z = Z[:self.M, :]
            else:
                raise NotImplementedError

        k.fixed = self.run_settings.get('fixed', False)
        self.m = GPflow.svgp.SVGP(self.X, self.Y, k, GPflow.likelihoods.Bernoulli(), Z.copy(), num_latent=1,
                                  minibatch_size=self.run_settings.get('minibatch_size', self.M))
        self.m.Z.fixed = self.run_settings.get('fixedZ', False)

    def setup_logger(self, verbose=None):
        h = pd.read_pickle(self.hist_path) if os.path.exists(self.hist_path) else None
        if h is not None:
            print("Resuming from %s..." % self.hist_path)
        tasks = [
            opt_tools.tasks.DisplayOptimisation(opt_tools.seq_exp_lin(1.1, 20)),
            opt_tools.tasks.GPflowLogOptimisation(opt_tools.seq_exp_lin(1.1, 20)),
            opt_tools.gpflow_tasks.GPflowBinClassTracker(self.Xt[:, :], self.Yt[:, :],
                                                         opt_tools.seq_exp_lin(1.1, 80, 3),
                                                         verbose=True, store_x="final_only",
                                                         store_x_columns=['model.kern.convrbf.basekern.lengthscales',
                                                                          'model.kern.convrbf.basekern.variance'],
                                                         old_hist=h),
            opt_tools.tasks.StoreOptimisationHistory(self.hist_path, itertools.count(0, 60), verbose=False)
        ]
        self.logger = opt_tools.GPflowOptimisationHelper(self.m, tasks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run mnist01 experiment.')
    parser.add_argument('-M', help="Number of inducing points", type=int, default=10)
    parser.add_argument('--fixed', '-f', help="Fix the model hyperparameters.", action="store_true", default=False)
    parser.add_argument('--fixedZ', '-fZ', help="Fix the inducing inputs.", action="store_true", default=False)
    parser.add_argument('--name', '-n', help="Experiment name appendage.", type=str, default=None)
    parser.add_argument('--learning-rate', '-l', help="Learning rate.", type=float, default=0.01)
    parser.add_argument('--minibatch-size', '-b', help="minibatch size.", type=int, default=100)
    parser.add_argument('--profile', help="Only run a quick profile of an iteration.", action="store_true",
                        default=False)
    parser.add_argument('--optimiser', help="Optimiser.", default="adam")
    parser.add_argument('--no-opt', help="Do not optimise.", action="store_true", default=False)
    parser.add_argument('--kernel', '-k', help="Kernel.")
    parser.add_argument('--Zinit', help="Inducing patches init.", default="random", type=str)
    parser.add_argument('--kernel-ard', help="Switch ARD on in the kernel.", default=False, action="store_true")
    args = parser.parse_args()

    # if GPflow.settings.dtypes.float_type is not tf.float32:
    #     raise RuntimeError("float_type must be float32, as set in gpflowrc.")

    run_settings = vars(args).copy()
    del run_settings['profile']
    del run_settings['no_opt']
    del run_settings['name']
    exp = Mnist01Experiment(name=args.name, M=args.M, run_settings=run_settings)

    if args.profile:
        print("Profiling an iteration...")
        exp.profile()
    else:
        print(exp.experiment_name)
        exp.setup()
        if not args.no_opt:
            exp.run()
