import itertools
import sys
import time
import warnings

import numpy as np
import pandas as pd


class OptimisationIterationEvent(object):
    def __init__(self, sequence, trigger="iter"):
        self._seq = sequence
        self._trigger = trigger
        self._next = next(self._seq) if self._seq is not None else np.inf

    def setup(self, logger):
        pass

    def _event_handler(self, logger, x, final):
        raise NotImplementedError

    def __call__(self, logger, x, final=False):
        if ((self._trigger == "iter" and logger._i >= self._next) or
                (self._trigger == "time" and logger._total_timer.elapsed_time >= self._next) or
                final):
            self._event_handler(logger, x, final)
            self._next = next(self._seq)
            for _ in range(1000000):
                if self._next < (logger._i if self._trigger == "iter" else logger._total_timer.elapsed_time):
                    self._next = next(self._seq)
                else:
                    break


class DisplayOptimisation(OptimisationIterationEvent):
    def __init__(self, sequence, trigger="iter"):
        OptimisationIterationEvent.__init__(self, sequence, trigger)
        self._last_disp = (0, 0.0)
        print("Starting at %s" % time.ctime())
        print("Iter\tfunc\t\tgrad\t\titer/s\tWall iter/s\tTimestamp")

    def setup(self, logger):
        self._last_disp = (logger._i, logger._opt_timer.elapsed_time, logger._total_timer.elapsed_time)

    def _event_handler(self, logger, x, final):
        if final:
            print("")
        f, g = logger._fg(x)
        iter_per_time = (logger._i - self._last_disp[0]) / (logger._opt_timer.elapsed_time - self._last_disp[1] + 1e-6)
        iter_per_tt = (logger._i - self._last_disp[0]) / (logger._total_timer.elapsed_time - self._last_disp[2] + 1e-6)
        sys.stdout.write("\r")
        sys.stdout.write("%i\t%e\t%e\t%6.2f\t%6.2f\t\t%s" %
                         (logger._i, f, np.linalg.norm(g), iter_per_time, iter_per_tt, time.ctime()))
        sys.stdout.flush()
        self._last_disp = (logger._i, logger._opt_timer.elapsed_time, logger._total_timer.elapsed_time)


class LogOptimisation(OptimisationIterationEvent):
    hist_name = "hist"

    def __init__(self, sequence, trigger="iter", old_hist=None, store_fullg=False, store_x=None, store_x_columns=None):
        """
        Log the optimisation history. Can also initialise the parent logger to a previously stored state by passing
        `old_hist`. The parent logger's iteration and timers will be set.
        :param sequence: Sequence of times when to log.
        :param trigger: Trigger type (time | iter)
        :param old_hist: History to initialise with (pandas DataFrame).
        :param store_fullg: Store the full gradient vector.
        :param store_x: Store the full parameter vector.
        """
        OptimisationIterationEvent.__init__(self, sequence, trigger)
        self._old_hist = old_hist
        self._store_fullg = store_fullg
        self._store_x = store_x
        self._store_x_columns = store_x_columns
        self.resume_from_hist = True

    def setup(self, logger):
        self._setup_logger(logger)

    def _get_hist(self, logger):
        return getattr(logger, self.hist_name)

    def _set_hist(self, logger, hist):
        return setattr(logger, self.hist_name, hist)

    def _get_columns(self, logger):
        return ['i', 't', 'tt', 'f', 'gnorm', 'g', 'x']

    def _setup_logger(self, logger):
        if self._old_hist is None:
            self._set_hist(logger, pd.DataFrame(columns=self._get_columns(logger)))
        else:
            self._set_hist(logger, self._old_hist)
            if self.resume_from_hist:
                hist = self._get_hist(logger)
                logger._i = hist.i.max()
                logger._opt_timer.add_time(hist.t.max())
                logger._total_timer.add_time(hist.tt.max())

    def _get_record(self, logger, x, f=None):
        if f is None:
            f, g = logger._fg(x)
        log_dict = dict(zip(
            self._get_hist(logger).columns,
            (logger._i, logger._opt_timer.elapsed_time, logger._total_timer.elapsed_time, f, np.linalg.norm(g),
             g if self._store_fullg else 0.0, x.copy() if self._store_x is not None else None)
        ))
        if logger._opt_options is not None:
            log_dict.update(logger._opt_options)
        return log_dict

    def _event_handler(self, logger, x, final, f=None):
        """
        _event_handler
        Adds the record from self._get_record to the hist object. If a record for the current iteration already exists,
        it is replaced.
        :param logger: Parent logger
        :param x: Current parameter value
        :param final: Indicator for final call
        :param f: ...
        :return: None
        """
        hist = self._get_hist(logger)
        if len(hist) > 0 and hist.iloc[-1, :].i == logger._i:
            hist = hist.iloc[:-1, :]

        if self._store_x == "final_only":
            if self._store_x_columns is None:
                hist.iloc[:, ['model.' in c for c in hist.columns]] = np.nan
            else:
                hist.iloc[:, [c not in self._store_x_columns and 'model.' in c for c in hist.columns]] = np.nan
        elif self._store_x not in [None, True]:
            raise ValueError("Unknown value for store_x: %s." % str(self._store_x))

        self._set_hist(logger, hist.append(self._get_record(logger, x), ignore_index=True))


class GPflowLogOptimisation(LogOptimisation):
    """
    Log the optimisation progress of a GPflow object, and restore it from history.
    """

    def _get_columns(self, logger):
        return ['i', 'feval', 't', 'tt', 'f', 'gnorm', 'g'] + list(logger.model.get_parameter_dict().keys())

    def _setup_logger(self, logger):
        """
        _setup_logger
        Properly resume from the history.
         - Set the model to the last stored parameters.
         - Check whether the function values align.
        :param logger: Parent logger
        :return: None
        """
        super(GPflowLogOptimisation, self)._setup_logger(logger)
        if self._old_hist is not None and self.resume_from_hist:
            # Get last record with parameters
            # params = self._old_hist.iloc[-1].filter(regex='(?!.*gpflow-opt*)model')
            params = self._old_hist.filter(regex='model*')
            params = params[~self._old_hist.acc.isnull()]

            # logger.model.set_parameter_dict(params.filter(regex="model.*").iloc[-1])
            logger.model.set_parameter_dict(params.filter(regex='(?!.*gpflow-opt*)model').iloc[-1])
            # Comment out the following line to disable restoring optimiser variables.
            logger.model.set_optimizer_variables_value(params.filter(regex='model\.gpflow-opt\.*').iloc[-1])
            f, _ = logger._fg(logger.model.get_free_state())
            hist = self._get_hist(logger)

            if not np.allclose(f, hist.iloc[-1].f):
                warnings.warn(
                    "Reloaded and stored function values don't match exactly: %f vs %f" % (f, hist.iloc[-1].f),
                    RuntimeWarning)

            logger.model.num_fevals = hist.iloc[-1].feval

    def _get_record(self, logger, x, f=None):
        if f is None:
            f, g = logger._fg(x)
        log_dict = dict(zip(
            self._get_hist(logger).columns[:7],
            (logger._i, logger.model.num_fevals, logger._opt_timer.elapsed_time, logger._total_timer.elapsed_time, f,
             np.linalg.norm(g), g if self._store_fullg else 0.0)
        ))
        if self._store_x is not None:
            log_dict.update(logger.model.get_samples_df(x[None, :].copy()).iloc[0, :].to_dict())
        if logger._opt_options is not None:
            log_dict.update(logger._opt_options)
        return log_dict


class StoreOptimisationHistory(OptimisationIterationEvent):
    def __init__(self, store_path, sequence, trigger="time", verbose=False, hist_name="hist"):
        """
        Stores the optimisation history present in the associated `logger` object.
        :param store_path: Path to store the history.
        :param sequence: Sequence of times when to store.
        :param trigger: Trigger type (time | iter)
        :param verbose: Display when history is stored.
        """
        OptimisationIterationEvent.__init__(self, sequence, trigger)
        self._store_path = store_path
        self._verbose = verbose
        self.hist_name = hist_name

    def _event_handler(self, logger, x, final):
        st = time.time()
        getattr(logger, self.hist_name).to_pickle(self._store_path)
        store_time = time.time() - st
        if self._verbose:
            print("")
            print("Stored history in %.2fs" % store_time)
        if store_time > 10:
            warnings.warn("Storing history is taking long (%.2fs)." % store_time)


class OptimisationTimeout(Exception):
    pass


class Timeout(OptimisationIterationEvent):
    """
    Timeout
    """

    def __init__(self, threshold, trigger="time"):
        """
        Triggers an OptimisationTimout exception when the total runtime exceeds the threshold.
        :param threshold: Maximum time / iterations before raising the exception.
        :param trigger: Chosen trigger (time | iter).
        """
        OptimisationIterationEvent.__init__(self, itertools.cycle([threshold]), trigger)
        self._triggered = False

    def _event_handler(self, logger, x, final):
        if not self._triggered and not final:
            self._triggered = True
            logger.finish(x)
            raise OptimisationTimeout()
