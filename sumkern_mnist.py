"""
choose-mnist.py
Try to automatically choose between convolutional or RBF structure.
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
from svconvgp.svsumgp import MeanFieldSVSumGP, FullSVSumGP


class ChooseMnistExperiment(exp_tools.MnistExperiment):
    def __init__(self, name=None, M=100, run_settings=None):
        exp_name = ("fullmnist-%s%s%i-%s" %
                    (run_settings["kernel1"], run_settings['kernel2'], M,
                     run_settings['vardist'])) if name is None else name
        super(ChooseMnistExperiment, self).__init__(exp_name)
        self.M = M
        self.run_settings = run_settings

    def setup_model(self):
        patch_size = self.run_settings['patch_size']
        # k1 = ckern.ConvRBF([28, 28], [patch_size, patch_size])  # Experiment was mistakenly run with no weighting
        if self.run_settings["kernel1"] == "wconv":
            k1 = ckern.WeightedConv(GPflow.kernels.RBF(25), [28, 28], [patch_size, patch_size])
        elif self.run_settings['kernel1'] == "conv":
            k1 = ckern.Conv(GPflow.kernels.RBF(25), [28, 28], [patch_size, patch_size])
        else:
            raise NotImplementedError("Unknown setting for `kernel1`.")

        if self.run_settings['kernel2'] == "rbf":
            k2 = GPflow.kernels.RBF(28 * 28) + GPflow.kernels.White(28 * 28)
        elif self.run_settings['kernel2'] == "poly":
            raise NotImplementedError("Look up what the poly was used in the paper Miguel sent you.")
        elif self.run_settings['kernel2'] == "rbfpoly":
            raise NotImplementedError("Look up what the poly was used in the paper Miguel sent you.")
        else:
            raise NotImplementedError("Unknown setting for `kernel2`.")
        k2.white.variance = 1e-2
        k1.fixed = self.run_settings.get('fixed', False)
        k2.fixed = self.run_settings.get('fixed', False)
        k1.basekern.variance = 1.0
        k2.rbf.variance = 1.1
        Z1 = k1.init_inducing(self.X, self.M, method=self.run_settings['Zinit'])
        Z2 = self.X[np.random.permutation(len(self.X))[:self.M], :]
        model_class = MeanFieldSVSumGP if run_settings["vardist"] == "mf" else FullSVSumGP
        self.m = model_class(self.X, self.Y, k1, k2, GPflow.likelihoods.MultiClass(10), Z1.copy(), Z2.copy(),
                             num_latent=10, minibatch_size=self.run_settings.get('minibatch_size', self.M))

    def setup_logger(self, verbose=False):
        h = pd.read_pickle(self.hist_path) if os.path.exists(self.hist_path) else None
        if h is not None:
            print("Resuming from %s..." % self.hist_path)
        tasks = [
            opt_tools.tasks.DisplayOptimisation(opt_tools.seq_exp_lin(1.1, 20)),
            opt_tools.tasks.GPflowLogOptimisation(opt_tools.seq_exp_lin(1.1, 20)),
            exp_tools.GPflowMultiClassificationTrackerLml(
                self.Xt[:, :], self.Yt[:, :], itertools.count(1800, 1800), trigger="time",
                verbose=True, store_x="final_only", store_x_columns=".*(variance|lengthscales)"),
            opt_tools.gpflow_tasks.GPflowMultiClassificationTracker(
                self.Xt[:, :], self.Yt[:, :], opt_tools.seq_exp_lin(1.5, 150, start_jump=30), trigger="time",
                verbose=True, store_x="final_only", store_x_columns=".*(variance|lengthscales)", old_hist=h),
            opt_tools.tasks.StoreOptimisationHistory(self.hist_path, opt_tools.seq_exp_lin(1.5, 600, start_jump=30),
                                                     trigger="time", verbose=False),
            opt_tools.tasks.Timeout(self.run_settings.get("timeout", np.inf))
        ]
        if self.run_settings.get("reset_optimiser", False):
            self.m.set_optimizer_variables_value(None)
        self.logger = opt_tools.GPflowOptimisationHelper(self.m, tasks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run MNIST experiment.')
    parser.add_argument('--fixed', '-f', help="Fix the model hyperparameters.", action="store_true", default=False)
    parser.add_argument('--name', '-n', help="Experiment name appendage.", type=str, default=None)
    parser.add_argument('--learning-rate', '-l', help="Learning rate.", type=str, default="0.001")
    parser.add_argument('--learning-rate-block-iters', type=int, default=3600,
                        help="How many iterations to use in a run with a single learning rate.")
    parser.add_argument('--profile', help="Only run a quick profile of an iteration.", action="store_true",
                        default=False)
    parser.add_argument('--no-opt', help="Do not optimise.", action="store_true", default=False)
    parser.add_argument('-M', help="Number of inducing points.", type=int, default=100)
    parser.add_argument('--minibatch-size', help="Size of the minibatch.", type=int, default=100)
    parser.add_argument('--benchmarks', action="store_true", default=False)
    parser.add_argument('--optimiser', '-o', type=str, default="adam")
    parser.add_argument('--reset-optimiser', '-r', action="store_true")
    parser.add_argument('--vardist', choices=["mf", "full"])
    parser.add_argument('--patch-size', type=int, default=5)
    parser.add_argument('--kernel1', '-k1', help="First kernel (conv | wconv)")
    parser.add_argument('--kernel2', '-k2', help="Second kernel (rbf | poly | rbfpoly)")
    parser.add_argument('--Zinit', help="Inducing patches init.", default="patches-unique", type=str)
    parser.add_argument('--lml', help="Compute log marginal likelihood.", default=False, action="store_true")
    args = parser.parse_args()

    # if GPflow.settings.dtypes.float_type is not tf.float32:
    #     raise RuntimeError("float_type must be float32, as set in gpflowrc.")

    run_settings = vars(args).copy()
    del run_settings['profile']
    del run_settings['no_opt']
    del run_settings['name']

    exp = ChooseMnistExperiment(name=args.name, M=args.M, run_settings=run_settings)

    if args.profile:
        print("Profiling an iteration...")
        exp.profile()
    elif not args.no_opt:
        lr_base = run_settings['learning_rate']
        while True:
            print(exp.experiment_name)
            i = pd.read_pickle(exp.hist_path).i.max() if os.path.exists(exp.hist_path) else 1.0
            b = args.learning_rate_block_iters
            print("learning rate: %s" % args.learning_rate)
            run_settings['learning_rate'] = eval(args.learning_rate)  # Can use i and b in learning_rate
            print(run_settings['learning_rate'], i)
            exp.setup()
            rndstate = np.random.randint(0, 1e9)
            exp.m.X.index_manager.rng = np.random.RandomState(rndstate)
            exp.m.Y.index_manager.rng = np.random.RandomState(rndstate)
            exp.run(maxiter=args.learning_rate_block_iters)

    if args.benchmarks:
        exp.setup()
        p = exp.m.predict_y(exp.Xt)[0]
        lpp = np.mean(np.log(p[np.arange(len(exp.Xt)), exp.Yt.flatten()]))
        acc = np.mean(p.argmax(1) == exp.Yt[:, 0])

        print("Accuracy: %f" % acc)
        print("Lpp     : %f" % lpp)

    if args.lml:
        exp.setup()
        lml = exp_tools.calculate_large_batch_lml(exp.m, args.minibatch_size, exp.m.X.shape[0] // args.minibatch_size,
                                                  progress=True)
        print(lml)
