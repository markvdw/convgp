import sys
import time
import numpy as np
import pandas as pd


class OptimisationIterationEvent(object):
    def __init__(self, func, sequence, trigger="iter"):
        self._func = func
        self._seq = sequence
        self._trigger = trigger
        self._next = next(self._seq) if self._seq is not None else np.inf

    def __call__(self, logger, x, fg=None, final=False):
        if (self._trigger == "iter" and logger._i >= self._next) or (
                        self._trigger == "time" and time.time() - logger._start_time >= self._next) or final:
            self._func(x)
            self._next = next(self._seq)


class OptimisationLogger(object):
    def __init__(self, f, disp_sequence=None, hist_sequence=None, g=None, chaincallback=None, store_fullg=False,
                 store_x=False, hist=None):
        # Options
        self._f = f
        self._g = g

        disp_event = OptimisationIterationEvent(self._disp_func,
                                                disp_sequence if disp_sequence is not None else seq_exp_lin(1.5, 1000),
                                                "iter")

        log_event = OptimisationIterationEvent(self._log_hist_func,
                                               hist_sequence if hist_sequence is not None else seq_exp_lin(1.5, 1000),
                                               "iter")

        self._events = [disp_event, log_event]

        self._chaincallback = chaincallback
        self._store_fullg = store_fullg
        self._store_x = store_x

        # Working variables
        self._last_disp = (0, time.time())
        self._i = 0
        self._start_time = time.time()

        # Public members

        if hist is None:
            self.hist = pd.DataFrame(columns=['i', 't', 'f', 'gnorm', 'g', 'x'])
        else:
            self._continue_from_hist(hist)

        print("Iter\tfunc\t\tgrad\t\titer/s\t\tTimestamp")

    def _fg(self, x):
        """
        Distinguish between separate functions for f and g or a single one, and call the appropriate ones.
        :param x:
        :return:
        """
        f = self._f(x)
        if type(f) is tuple or type(f) is list:
            return f
        elif self._g is not None:
            return f, self._g(x)
        else:
            return f, 0.0

    def _disp_func(self, x):
        # Print and log
        f, g = self._fg(x)
        iter_per_time = (self._i - self._last_disp[0]) / (time.time() - self._last_disp[1])
        sys.stdout.write("\r")
        sys.stdout.write(
            "%i\t%e\t%e\t%f\t%s\t" % (self._i, f, np.linalg.norm(g), iter_per_time, time.ctime()))
        sys.stdout.flush()
        self._last_disp = (self._i, time.time())

    def _log_hist_func(self, x, f=None):
        if len(self.hist) > 0 and self.hist.iloc[-1, :].i == self._i:
            return
        if f is None:
            f, g = self._fg(x)
        self.hist = self.hist.append(dict(zip(self.hist.columns,
                                              (self._i, time.time() - self._start_time, f, np.linalg.norm(g),
                                               g if self._store_fullg else 0.0,
                                               x.copy() if self._store_x else None))),
                                     ignore_index=True)

    def callback(self, x, final=False):
        self._i += 1
        for event in self._events:
            event(self, x, final=final)

        if self._chaincallback is not None:
            self._chaincallback(x)

    def _continue_from_hist(self, hist):
        """
        Reset the timer to continue from where hist left off.
        """
        self.hist = hist
        if len(self.hist) > 0:
            self._start_time = time.time() - self.hist.iloc[-1, :].t
            self._i = self.hist.iloc[-1, :].i

    def finish(self, x):
        for event in self._events:
            event._func(x)


class OptimisationHelper(OptimisationLogger):
    def __init__(self, f, disp_sequence=None, hist_sequence=None, store_sequence=None, g=None, chaincallback=None,
                 store_fullg=False, store_x=False, timeout=np.inf, store_path=None, store_trigger="time", hist=None,
                 verbose=False):
        """
        :param f: Returns objective function value or list/tuple of objfunc + gradient
        :param disp_sequence: Sequence of iterations when to display to screen
        :param hist_sequence: Sequence of iterations when to record in history
        :param store_sequence: Sequence of times when to store history to disk
        :param g:
        :param chaincallback: Not implemented yet
        :param store_fullg: Keep history of the full gradient evaluation
        :param store_x:
        :param timeout:
        :param store_path: File where history is stored.
        """
        super(OptimisationHelper, self).__init__(f, disp_sequence, hist_sequence, g, chaincallback, store_fullg,
                                                 store_x, hist)
        self._verbose = verbose
        if (hist is not None) and (len(hist) > 0):
            timeout = timeout + hist.iloc[-1, :].t

        self._timeout_event = OptimisationIterationEvent(self._timeout_func, seq_exp_lin(1.0, timeout, timeout), "time")
        store_event = OptimisationIterationEvent(self._store_func, store_sequence, store_trigger)
        self._events.append(store_event)

        self._store_path = store_path
        if store_sequence is not None and store_path is None:
            raise ValueError("Need a `store_path` to store history file.")

    def _timeout_func(self, x):
        self.finish(x)
        raise KeyboardInterrupt

    def _store_func(self, x):
        st = time.time()
        self.hist.to_pickle(self._store_path)
        if self._verbose:
            print("")
            print("Stored history in %.2fs" % (time.time() - st))

    def store_hist(self):
        self.hist.to_pickle(self._store_path)

    def callback(self, x, final=False):
        super(OptimisationHelper, self).callback(x, final)
        self._timeout_event(self, x)


class GPflowOptimisationHelper(OptimisationHelper):
    def __init__(self, model, disp_sequence=None, hist_sequence=None, store_sequence=None,
                 chaincallback=None, store_fullg=False, store_x=False, timeout=np.inf, store_path=None,
                 store_trigger="time", hist=None, verbose=False):
        if hist is None:
            hist = pd.DataFrame(columns=['i', 't', 'f', 'gnorm', 'g'] + list(model.get_parameter_dict().keys()))
        super(GPflowOptimisationHelper, self).__init__(None, disp_sequence, hist_sequence, store_sequence, None,
                                                       chaincallback, store_fullg, store_x, timeout, store_path,
                                                       store_trigger, hist, verbose)
        self.model = model

    def _fg(self, x):
        return self.model._objective(x)

    def _log_hist_func(self, x, f=None):
        if len(self.hist) > 0 and self.hist.iloc[-1, :].i == self._i:
            return
        if f is None:
            f, g = self._fg(x)
        log_dict = dict(zip(self.hist.columns[:5], (
            self._i, time.time() - self._start_time, f, np.linalg.norm(g), g if self._store_fullg else 0.0)))
        log_dict.update(self.model.get_samples_df(x[None, :].copy()).iloc[0, :].to_dict())

        self.hist = self.hist.append(log_dict, ignore_index=True)

    @property
    def param_hist(self):
        return self.hist.loc[:, ["model" in v for v in self.hist.columns]]


def seq_exp_lin(growth, max, start=1.0, start_jump=1.0):
    gap = start_jump
    last = start - start_jump
    while 1:
        yield gap + last
        last = last + gap
        gap = min(gap * growth, max)
