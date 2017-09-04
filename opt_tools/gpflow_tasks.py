"""
Custom opt_tools tasks. Eventually, I think these can be merged into the main package.
"""
import time

import numpy as np

import opt_tools


class GPflowBenchmarkTrackerBase(opt_tools.tasks.GPflowLogOptimisation):
    def __init__(self, test_X, test_Y, sequence, trigger="iter", old_hist=None, store_fullg=False, store_x=None,
                 store_x_columns=None, verbose=False):
        opt_tools.tasks.GPflowLogOptimisation.__init__(self, sequence, trigger, old_hist, store_fullg, store_x,
                                                       store_x_columns)
        self.test_X = test_X
        self.test_Y = test_Y
        self.verbose = verbose

    def _get_record(self, logger, x, f=None):
        log_dict = super(GPflowBenchmarkTrackerBase, self)._get_record(logger, x, f)
        log_dict.update(logger.model.get_optimizer_variables()[0])
        return log_dict


class GPflowRegressionTracker(GPflowBenchmarkTrackerBase):
    def _get_columns(self, logger):
        return super(GPflowRegressionTracker, self)._get_columns(logger) + ['rmse', 'nlpp', 'pred_time']

    def _get_record(self, logger, x, f=None):
        st = time.time()
        log_dict = super(GPflowRegressionTracker, self)._get_record(logger, x, f)
        logger.model.set_state(x)

        pY, pYv = logger.model.predict_y(self.test_X)
        rmse = np.mean((pY - self.test_Y) ** 2.0) ** 0.5
        nlpp = -np.mean(-0.5 * np.log(2 * np.pi * pYv) - 0.5 * (self.test_Y - pY) ** 2.0 / pYv)
        log_dict.update({'rmse': rmse, 'nlpp': nlpp, 'pred_time': time.time() - st})

        if self.verbose:
            print("Benchmarks took %.2fs." % (time.time() - st))

        return log_dict


class GPflowBinClassTracker(GPflowBenchmarkTrackerBase):
    def _get_columns(self, logger):
        return super(GPflowBinClassTracker, self)._get_columns(logger) + ['acc', 'nlpp']

    def _get_record(self, logger, x, f=None):
        st = time.time()
        log_dict = super(GPflowBinClassTracker, self)._get_record(logger, x, f)
        logger.model.set_state(x)

        p, var = logger.model.predict_y(self.test_X)
        acc = ((p > 0.5).astype('float') == self.test_Y).mean()
        nlpp = -np.mean(np.log(p) * self.test_Y + np.log(1 - p) * (1 - self.test_Y))
        log_dict.update({'acc': acc, 'err': 1 - acc, 'nlpp': nlpp})

        if self.verbose:
            print("Benchmarks took %.2fs (err: %.4f, nlpp: %.4f)." % (time.time() - st, 1 - acc, nlpp))

        return log_dict


class GPflowMultiClassificationTracker(GPflowBenchmarkTrackerBase):
    def _get_columns(self, logger):
        return super(GPflowMultiClassificationTracker, self)._get_columns(logger) + ['acc', 'nlpp']

    def _get_record(self, logger, x, f=None):
        st = time.time()
        log_dict = super(GPflowMultiClassificationTracker, self)._get_record(logger, x, f)
        logger.model.set_state(x)

        pred_batch_size = 1000 if not hasattr(self, "pred_batch_size") else self.pred_batch_size

        p = np.vstack([logger.model.predict_y(self.test_X[n * pred_batch_size:(n + 1) * pred_batch_size, :])[0]
                       for n in range(-(-len(self.test_X) // pred_batch_size))])
        assert len(p) == len(self.test_X)
        # acc = ((p > 0.5).astype('float') == self.test_Y).mean()
        acc = (np.argmax(p, 1) == self.test_Y[:, 0]).mean()
        pcorrect = p[self.test_Y == np.arange(0, 10)[None, :]]
        nlpp = -np.mean(np.log(pcorrect))

        log_dict.update({'acc': acc, 'err': 1 - acc, 'nlpp': nlpp})

        if self.verbose:
            print("Benchmarks took %.2fs (err: %.4f, nlpp: %.4f)." % (time.time() - st, 1 - acc, nlpp))

        return log_dict
