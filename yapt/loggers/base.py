"""
https://github.com/PyTorchLightning/pytorch-lightning/blob/master/pytorch_lightning/loggers/base.py
"""
import torch
from abc import ABC, abstractmethod
from typing import Union, Optional, Dict, Iterable, Any, Callable, List
from argparse import Namespace

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
    @abstractmethod
    def experiment(self) -> Any:
        """Return the experiment object associated with this logger"""

    @abstractmethod
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Record metrics.
        Args:
            metrics: Dictionary with metric names as keys and measured quantities as values
            step: Step number at which the metrics should be recorded
        """

    @staticmethod
    def _convert_params(params: Union[Dict[str, Any], Namespace]) -> Dict[str, Any]:
        # in case converting from namespace
        if isinstance(params, Namespace):
            params = vars(params)

        if params is None:
            params = {}

        return params

    @staticmethod
    def _flatten_dict(params: Dict[str, Any], delimiter: str = '/') -> Dict[str, Any]:
        """Flatten hierarchical dict e.g. {'a': {'b': 'c'}} -> {'a/b': 'c'}.
        Args:
            params: Dictionary contains hparams
            delimiter: Delimiter to express the hierarchy. Defaults to '/'.
        Returns:
            Flatten dict.
        Examples:
            >>> LightningLoggerBase._flatten_dict({'a': {'b': 'c'}})
            {'a/b': 'c'}
            >>> LightningLoggerBase._flatten_dict({'a': {'b': 123}})
            {'a/b': 123}
        """

        def _dict_generator(input_dict, prefixes=None):
            prefixes = prefixes[:] if prefixes else []
            if isinstance(input_dict, dict):
                for key, value in input_dict.items():
                    if isinstance(value, (dict, Namespace)):
                        value = vars(value) if isinstance(value, Namespace) else value
                        for d in _dict_generator(value, prefixes + [key]):
                            yield d
                    else:
                        yield prefixes + [key, value if value is not None else str(None)]
            else:
                yield prefixes + [input_dict if input_dict is None else str(input_dict)]

        return {delimiter.join(keys): val for *keys, val in _dict_generator(params)}

    @staticmethod
    def _sanitize_params(params: Dict[str, Any]) -> Dict[str, Any]:
        """Returns params with non-primitvies converted to strings for logging
        >>> params = {"float": 0.3,
        ...           "int": 1,
        ...           "string": "abc",
        ...           "bool": True,
        ...           "list": [1, 2, 3],
        ...           "namespace": Namespace(foo=3),
        ...           "layer": torch.nn.BatchNorm1d}
        >>> import pprint
        >>> pprint.pprint(LightningLoggerBase._sanitize_params(params))  # doctest: +NORMALIZE_WHITESPACE
        {'bool': True,
         'float': 0.3,
         'int': 1,
         'layer': "<class 'torch.nn.modules.batchnorm.BatchNorm1d'>",
         'list': '[1, 2, 3]',
         'namespace': 'Namespace(foo=3)',
         'string': 'abc'}
        """
        return {k: v if type(v) in [bool, int, float, str, torch.Tensor] else str(v) for k, v in params.items()}

    @abstractmethod
    def log_hyperparams(self, params: Namespace):
        """Record hyperparameters.
        Args:
            params: argparse.Namespace containing the hyperparameters
        """

    def save(self) -> None:
        """Save log data."""
        pass

    def finalize(self, status: str) -> None:
        """Do any processing that is necessary to finalize an experiment.
        Args:
            status: Status that the experiment finished with (e.g. success, failed, aborted)
        """
        pass

    def close(self) -> None:
        """Do any cleanup that is necessary to close an experiment."""
        pass

    @property
    def rank(self) -> int:
        """Process rank. In general, metrics should only be logged by the process with rank 0."""
        return self._rank

    @rank.setter
    def rank(self, value: int) -> None:
        """Set the process rank."""
        self._rank = value

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the experiment name."""

    @property
    @abstractmethod
    def version(self) -> Union[int, str]:
        """Return the experiment version."""


class LoggerList():

    """Module wrapper for a list of loggers objects"""

    def __init__(self, loggers: list):
        assert isinstance(loggers, (list, tuple)), \
            "loggers should be a list/tuple"

        for idx, logger in enumerate(loggers):
            assert isinstance(logger, LoggerBase), \
                "%s idx is not a logger!" % idx
        self._loggers = loggers

    def __getitem__(self, index):
        return self._loggers[index]

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

    def log_artifact(self, artifact: str, destination: Optional[str] = None) -> None:
        """Save an artifact (file) in storage (if the logger allows it)

        Args:
            artifact: A path to the file in local filesystem.
            destination: Optional default None. A destination path.
                If None is passed, an artifact file name will be used.
        """
        for logger in self._loggers:
            log_fn = getattr(logger, "log_artifact", None)
            if callable(log_fn):
                log_fn(artifact, destination)

    def log_image(self, log_name: str, image: Union[str, Any], step: Optional[int] = None) -> None:
        """Log image data if the logger allows it.

        Args:
            log_name: The name of log, i.e. bboxes, visualisations, sample_images.
            image (str|PIL.Image|matplotlib.figure.Figure): The value of the log (data-point).
                Can be one of the following types: PIL image, matplotlib.figure.Figure, path to image file (str)
            step: Step number at which the metrics should be recorded, must be strictly increasing
        """
        for logger in self._loggers:
            log_fn = getattr(logger, "log_image", None)
            if callable(log_fn):
                log_fn(log_name, image, step)

    def log_chart(self, log_name: str, image: Union[str, Any], step: Optional[int] = None) -> None:
        """Logs charts from matplotlib, plotly, bokeh, and altair to neptune.

        Plotly, Bokeh, and Altair charts are converted to interactive HTML objects and then uploaded to Neptune
        as an artifact with path charts/{name}.html.

        Matplotlib figures are converted optionally. If plotly is installed, matplotlib figures are converted
        to plotly figures and then converted to interactive HTML and uploaded to Neptune as an artifact with
        path charts/{name}.html. If plotly is not installed, matplotlib figures are converted to PNG images
        and uploaded to Neptune as an artifact with path charts/{name}.png

        Args:
            log_name (:obj:`str`):
                | Name of the chart (without extension) that will be used as a part of artifact's destination.
            chart (:obj:`matplotlib` or :obj:`plotly` Figure):
                | Figure from `matplotlib` or `plotly`. If you want to use global figure from `matplotlib`, you
                  can also pass reference to `matplotlib.pyplot` module.
        """
        for logger in self._loggers:
            log_fn = getattr(logger, "log_chart", None)
            if callable(log_fn):
                log_fn(log_name, image)

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

    def __getitem__(self, key):
        return self._loggers[key]

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

    def log_artifact(self, artifact: str, destination: Optional[str] = None) -> None:
        """Save an artifact (file) in storage (if the logger allows it)

        Args:
            artifact: A path to the file in local filesystem.
            destination: Optional default None. A destination path.
                If None is passed, an artifact file name will be used.
        """
        for key, logger in self._loggers.items():
            log_fn = getattr(logger, "log_artifact", None)
            if callable(log_fn):
                log_fn(artifact, destination)

    def log_image(self, log_name: str, image: Union[str, Any], step: Optional[int] = None) -> None:
        """Log image data if the logger allows it.

        Args:
            log_name: The name of log, i.e. bboxes, visualisations, sample_images.
            image (str|PIL.Image|matplotlib.figure.Figure): The value of the log (data-point).
                Can be one of the following types: PIL image, matplotlib.figure.Figure, path to image file (str)
            step: Step number at which the metrics should be recorded, must be strictly increasing
        """
        for key, logger in self._loggers.items():
            log_fn = getattr(logger, "log_image", None)
            if callable(log_fn):
                log_fn(log_name, image, step)

    def log_chart(self, log_name: str, image: Union[str, Any], step: Optional[int] = None) -> None:
        """Logs charts from matplotlib, plotly, bokeh, and altair to neptune.

        Plotly, Bokeh, and Altair charts are converted to interactive HTML objects and then uploaded to Neptune
        as an artifact with path charts/{name}.html.

        Matplotlib figures are converted optionally. If plotly is installed, matplotlib figures are converted
        to plotly figures and then converted to interactive HTML and uploaded to Neptune as an artifact with
        path charts/{name}.html. If plotly is not installed, matplotlib figures are converted to PNG images
        and uploaded to Neptune as an artifact with path charts/{name}.png

        Args:
            log_name (:obj:`str`):
                | Name of the chart (without extension) that will be used as a part of artifact's destination.
            chart (:obj:`matplotlib` or :obj:`plotly` Figure):
                | Figure from `matplotlib` or `plotly`. If you want to use global figure from `matplotlib`, you
                  can also pass reference to `matplotlib.pyplot` module.
        """
        for key, logger in self._loggers.items():
            log_fn = getattr(logger, "log_chart", None)
            if callable(log_fn):
                log_fn(log_name, image)

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


