"""
vanilla_rectangles.py
Run the convolutional GP on rectangles, using standard binary classification
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


class RectanglesExperiment(exp_tools.RectanglesExperiment):
    def __init__(self, name=None, M=6, run_settings=None):
        super(RectanglesExperiment, self).__init__("vanilla-rectangles%i" % M if name is None else name)
        self.test_slice = np.s_[:3000]
        self.run_settings = run_settings if run_settings is not None else {}
        self.M = M

    def setup_model(self):
        if self.run_settings['kernel'] == "rbf":
            k = GPflow.kernels.RBF(28 * 28)
            Z = self.X[np.random.permutation(len(self.X))[:self.M], :]
            k.lengthscales = 3.0
        elif self.run_settings['kernel'] == "conv":
            k = ckern.ConvRBF([28, 28], [3, 3]) + GPflow.kernels.White(1, 1e-3)
            # Z = np.zeros((1, k.convrbf.patch_len))
            # for x in np.split(self.X, len(self.X) // 100):
            #     patches = k.convrbf.compute_patches(x).reshape(-1, k.convrbf.patch_len)
            #     Z = np.vstack({tuple(row) for row in np.vstack((Z, patches))})
            #
            # Z = Z[:self.M, :]
            Z = np.random.rand(self.M, 9)

        k.fixed = self.run_settings.get('fixed', False)
        self.m = GPflow.svgp.SVGP(self.X, self.Y, k, GPflow.likelihoods.Bernoulli(), Z.copy(), num_latent=1,
                                  minibatch_size=self.run_settings.get('minibatch_size', self.M))

    def setup_logger(self, verbose=None):
        h = pd.read_pickle(self.hist_path) if os.path.exists(self.hist_path) else None
        if h is not None:
            print("Resuming from %s..." % self.hist_path)
        tasks = [
            opt_tools.tasks.DisplayOptimisation(opt_tools.seq_exp_lin(1.1, 20)),
            opt_tools.tasks.GPflowLogOptimisation(opt_tools.seq_exp_lin(1.1, 20)),
            opt_tools.gpflow_tasks.GPflowBinClassTracker(self.Xt[self.test_slice, :], self.Yt[self.test_slice, :],
                                                         opt_tools.seq_exp_lin(1.1, 80, 3),
                                                         verbose=True, store_x="final_only", old_hist=h),
            opt_tools.tasks.StoreOptimisationHistory(self.hist_path, itertools.count(0, 60),
                                                     verbose=False)
        ]
        self.logger = opt_tools.GPflowOptimisationHelper(self.m, tasks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run rectangles experiment.')
    parser.add_argument('-M', help="Number of inducing points", type=int, default=10)
    parser.add_argument('--fixed', '-f', help="Fix the model hyperparameters.", action="store_true", default=False)
    parser.add_argument('--name', '-n', help="Experiment name appendage.", type=str, default=None)
    parser.add_argument('--learning-rate', '-l', help="Learning rate.", type=float, default=0.01)
    parser.add_argument('--minibatch-size', '-b', help="minibatch size.", type=int, default=100)
    parser.add_argument('--profile', help="Only run a quick profile of an iteration.", action="store_true",
                        default=False)
    parser.add_argument('--optimiser', help="Optimiser.", default="adam")
    parser.add_argument('--no-opt', help="Do not optimise.", action="store_true", default=False)
    parser.add_argument('--kernel', help="Kernel.", default="conv")
    args = parser.parse_args()

    # if GPflow.settings.dtypes.float_type is not tf.float32:
    #     raise RuntimeError("float_type must be float32, as set in gpflowrc.")

    run_settings = vars(args).copy()
    del run_settings['profile']
    del run_settings['no_opt']
    del run_settings['name']
    exp = RectanglesExperiment(name=args.name, M=args.M, run_settings=run_settings)

    if args.profile:
        print("Profiling an iteration...")
        exp.profile()
    else:
        print(exp.experiment_name)
        exp.setup()
        if not args.no_opt:
            exp.run()
