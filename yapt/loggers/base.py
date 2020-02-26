"""
https://github.com/PyTorchLightning/pytorch-lightning/blob/master/pytorch_lightning/loggers/base.py
"""
from abc import ABC
from functools import wraps


def rank_zero_only(fn):
    """Decorate a logger method to run it only on the process with rank 0.
    :param fn: Function to decorate
    """

    @wraps(fn)
    def wrapped_fn(self, *args, **kwargs):
        if self.rank == 0:
            fn(self, *args, **kwargs)

    return wrapped_fn


class LoggerBase(ABC):
    """Base class for experiment loggers."""

    def __init__(self):
        self._rank = 0

    @property
    def experiment(self):
        raise NotImplementedError()

    def log_metrics(self, metrics, step):
        """Record metrics.
        :param float metric: Dictionary with metric names as keys and measured quanties as values
        :param int|None step: Step number at which the metrics should be recorded
        """
        raise NotImplementedError()

    def log_hyperparams(self, params):
        """Record hyperparameters.
        :param params: argparse.Namespace containing the hyperparameters
        """
        raise NotImplementedError()

    def save(self):
        """Save log data."""

    def finalize(self, status):
        """Do any processing that is necessary to finalize an experiment.
        :param status: Status that the experiment finished with (e.g. success, failed, aborted)
        """

    def close(self):
        """Do any cleanup that is necessary to close an experiment."""

    @property
    def rank(self):
        """Process rank. In general, metrics should only be logged by the process with rank 0."""
        return self._rank

    @rank.setter
    def rank(self, value):
        """Set the process rank."""
        self._rank = value

    @property
    def name(self):
        """Return the experiment name."""
        raise NotImplementedError("Sub-classes must provide a name property")

    @property
    def version(self):
        """Return the experiment version."""
        raise NotImplementedError("Sub-classes must provide a version property")


class LoggerList():

    """Module wrapper for a list of loggers objects"""

    def __init__(self, loggers: list):
        assert isinstance(loggers, (list, tuple)), \
            "loggers should be a list/tuple"

        for idx, logger in enumerate(loggers):
            assert isinstance(logger, LoggerBase), \
                "%s idx is not a logger!" % idx
        self._loggers = loggers

    @property
    def loggers(self):
        return self._loggers

    def log_metrics(self, metrics, step):
        """Record metrics.
        :param float metric: Dictionary with metric names as keys and measured quanties as values
        :param int|None step: Step number at which the metrics should be recorded
        """
        for logger in self._loggers:
            logger.log_metrics(metrics, step)

    def log_hyperparams(self, params):
        """Record hyperparameters.
        :param params: argparse.Namespace containing the hyperparameters
        """
        for logger in self._loggers:
            logger.log_hyperparams(params)

    def save(self):
        """Save log data."""
        for logger in self._loggers:
            logger.save()

    def finalize(self, status):
        """Do any processing that is necessary to finalize an experiment.
        :param status: Status that the experiment finished with (e.g. success, failed, aborted)
        """
        for logger in self._loggers:
            logger.finalize(status)

    def close(self):
        """Do any cleanup that is necessary to close an experiment."""
        for logger in self._loggers:
            logger.close()

    @property
    def rank(self):
        """Process rank. In general, metrics should only be logged by the process with rank 0."""
        ranks = []
        for logger in self._loggers:
            ranks.append(logger._rank)
        return ranks

    @rank.setter
    def rank(self, value):
        """Set the process rank."""
        for logger in self._loggers:
            logger._rank = value

    @property
    def name(self):
        """Return the experiment name."""
        names = []
        for logger in self._loggers:
            names.append(logger.name)
        return names

    @property
    def version(self):
        """Return the experiment version."""
        versions = []
        for logger in self._loggers:
            versions.append(logger.version)
        return versions


class LoggerDict():

    """Module wrapper for a dict of loggers objects"""

    def __init__(self, loggers: dict):
        assert isinstance(loggers, dict), "loggers should be a dict"
        for key, logger in loggers.items():
            assert isinstance(logger, LoggerBase), \
                "%s key is not a logger!" % key
        self._loggers = loggers

    @property
    def loggers(self):
        return self._loggers

    def log_metric(self, metric_name, metric_value, step):
        """Record metric.
        :param string metric_name
        :param float metric_val
        :param int|None step: Step number at which the metrics should be recorded
        """
        for key, logger in self._loggers.items():
            logger.log_metric(metric_name, metric_value, step)

    def log_metrics(self, metrics, step):
        """Record metrics.
        :param float metric: Dictionary with metric names as keys and measured quanties as values
        :param int|None step: Step number at which the metrics should be recorded
        """
        for key, logger in self._loggers.items():
            logger.log_metrics(metrics, step)

    def log_hyperparams(self, params):
        """Record hyperparameters.
        :param params: argparse.Namespace containing the hyperparameters
        """
        for key, logger in self._loggers.items():
            logger.log_hyperparams(params)

    def save(self):
        """Save log data."""
        for key, logger in self._loggers.items():
            logger.save()

    def finalize(self, status):
        """Do any processing that is necessary to finalize an experiment.
        :param status: Status that the experiment finished with (e.g. success, failed, aborted)
        """
        for key, logger in self._loggers.items():
            logger.finalize(status)

    def close(self):
        """Do any cleanup that is necessary to close an experiment."""
        for key, logger in self._loggers.items():
            logger.close()

    @property
    def rank(self):
        """Process rank. In general, metrics should only be logged by the process with rank 0."""
        ranks = {}
        for key, logger in self._loggers.items():
            ranks[key] = logger._rank
        return ranks

    @rank.setter
    def rank(self, value):
        """Set the process rank."""
        for key, logger in self._loggers.items():
            logger._rank = value

    @property
    def name(self):
        """Return the experiment name."""
        names = {}
        for key, logger in self._loggers.items():
            names[key] = logger.name
        return names

    @property
    def version(self):
        """Return the experiment version."""
        versions = []
        for key, logger in self._loggers.items():
            versions[key] = logger.version
        return versions


