import contextlib
import os
import sys

import jug
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import GPflow
import GPflow.minibatch as mb
import opt_tools


@jug.TaskGenerator
def jugrun_experiment(exp):
    print("Running %s..." % exp.experiment_name)
    exp.setup()
    try:
        exp.run()
    except opt_tools.OptimisationTimeout:
        print("Timeout")


@contextlib.contextmanager
def suppress_print():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


def load_mnist():
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    X = np.vstack((mnist.train.images.astype(float), mnist.validation.images.astype('float')))
    Y = np.vstack((np.argmax(mnist.train.labels, 1)[:, None],
                   np.argmax(mnist.validation.labels, 1)[:, None]))
    Xt = mnist.test.images.astype(float)
    Yt = np.argmax(mnist.test.labels, 1)[:, None]
    return X, Y, Xt, Yt


class ExperimentBase(object):
    def __init__(self, name):
        self.experiment_name = name
        self.m = None
        self.logger = None
        self.X = None
        self.Y = None
        self.Xt = None
        self.Yt = None
        self.run_settings = {}

    def setup_dataset(self, verbose=False):
        raise NotImplementedError

    def setup_model(self):
        raise NotImplementedError

    def setup_logger(self, verbose=False):
        raise NotImplementedError

    def setup(self, verbose=False):
        """
        setup
        Setup logger, model and anything else that isn't picklable.
        :return:
        """
        self.setup_dataset(verbose)
        self.setup_model()
        self.setup_logger(verbose)
        return self.m, self.logger

    def run(self, maxiter=np.inf):
        optimiser = self.run_settings.get("optimiser", "adam")
        if optimiser == "adam":
            opt_method = tf.train.AdamOptimizer(self.run_settings['learning_rate'])
        elif optimiser == "rmsprop":
            opt_method = tf.train.RMSPropOptimizer(self.run_settings['learning_rate'])
        else:
            opt_method = optimiser

        self.opt_method = opt_method

        try:
            return self.logger.optimize(method=opt_method, maxiter=maxiter, opt_options=self.run_settings)
        finally:
            self.logger.finish(self.m.get_free_state())

    def profile(self):
        """
        profile
        Run a few iterations and dump the timeline.
        :return:
        """
        s = GPflow.settings.get_settings()
        s.profiling.dump_timeline = True
        s.profiling.output_file_name = "./trace_" + self.experiment_name
        with GPflow.settings.temp_settings(s):
            self.m._compile()
            self.m._objective(self.m.get_free_state())
            self.m._objective(self.m.get_free_state())
            self.m._objective(self.m.get_free_state())

    def load_results(self):
        return pd.read_pickle(self.hist_path)

    @property
    def base_filename(self):
        return os.path.join('.', 'results', self.experiment_name)

    @property
    def hist_path(self):
        return self.base_filename + '_hist.pkl'

    @property
    def param_path(self):
        return self.base_filename + '_params.pkl'

    def __jug_hash__(self):
        from jug.hash import hash_one
        return hash_one(self.experiment_name)


class CifarExperiment(ExperimentBase):
    def setup_dataset(self, verbose=False):
        d = np.load('./datasets/cifar10.npz')
        self.X = (d['X'] / 255.0).reshape(50000, 3, 32, 32).swapaxes(1, 3).reshape(50000, -1)
        self.Y = d['Y'].astype('int64')
        self.Xt = (d['Xt'] / 255.0).reshape(10000, 3, 32, 32).swapaxes(1, 3).reshape(10000, -1)
        self.Yt = d['Yt'].astype('int64')

    def img_plot(self, i):
        import matplotlib.pyplot as plt
        plt.imshow(self.X[i, :].reshape(3, 32, 32).transpose([1, 2, 0]))


class MnistExperiment(ExperimentBase):
    def setup_dataset(self, verbose=False):
        with suppress_print():
            self.X, self.Y, self.Xt, self.Yt = load_mnist()


class RectanglesImageExperiment(ExperimentBase):
    def setup_dataset(self, verbose=False):
        d = np.load('datasets/rectangles_im.npz')
        self.X, self.Y, self.Xt, self.Yt = d['X'], d['Y'], d['Xtest'], d['Ytest']


def calculate_large_batch_lml(m, minibatch_size, batches, progress=False):
    """
    This does not work properly yet, presumably because it changes the state (w.r.t. _parent) of the model.
    """
    assert type(batches) == int, "`batches` must be an integer."
    old_mbX = m.X
    old_mbY = m.Y
    m.X = mb.MinibatchData(m.X.value, minibatch_size,
                           batch_manager=mb.SequenceIndices(minibatch_size, m.X.value.shape[0]))
    m.Y = mb.MinibatchData(m.Y.value, minibatch_size,
                           batch_manager=mb.SequenceIndices(minibatch_size, m.X.value.shape[0]))
    m._kill_autoflow()

    batch_lmls = []
    if progress:
        from tqdm import tqdm
        for _ in tqdm(range(batches)):
            batch_lmls.append(m.compute_log_likelihood())
    else:
        for _ in range(batches):
            batch_lmls.append(m.compute_log_likelihood())

    m.X = old_mbX
    m.Y = old_mbY

    m._kill_autoflow()

    return np.mean(batch_lmls)


class CalculateFullLMLMixin(object):
    def _get_record(self, logger, x, f=None):
        log_dict = super(CalculateFullLMLMixin, self)._get_record(logger, x, f)
        model = logger.model
        minibatch_size = logger.model.X.index_manager.minibatch_size
        lml = calculate_large_batch_lml(model, minibatch_size, model.X.shape[0] // minibatch_size, True)
        print("full lml: %f" % lml)
        log_dict.update({"lml": lml})
        return log_dict


class GPflowMultiClassificationTrackerLml(CalculateFullLMLMixin,
                                          opt_tools.gpflow_tasks.GPflowMultiClassificationTracker):
    pass
