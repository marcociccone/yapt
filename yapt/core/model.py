import logging
import torch.nn as nn

from abc import ABC, abstractmethod
from collections import OrderedDict

from yapt import _logger
from yapt import BaseTrainer
from yapt.loggers.base import LoggerDict
from yapt.utils.trainer_utils import (get_optimizer, get_scheduler_optimizer,
                                      detach_dict, to_device)
from yapt.utils.utils import (call_counter, warning_not_implemented,
                              get_maybe_missing_args, add_key_dict_prefix,
                              is_list, is_scalar, is_dict, recursive_keys)


def is_pickable(obj):
    non_pickable = (LoggerDict, logging.Logger, BaseTrainer)
    return not isinstance(obj, non_pickable)


class BaseModel(ABC, nn.Module):

    """Docstring for MyClass. """

    def __init__(self, args, logger=None, device='cpu', **kwargs):
        super().__init__(**kwargs)

        self.console_log = _logger

        self.args = args
        self.logger = logger
        self.device = device

        self._best_epoch = -1
        self._best_epoch_score = 0
        self._beaten_epochs = 0
        self._best_stats = []
        self._early_stop = False

        self._epoch = 0
        self._global_step = 0
        self._train_step = 0
        self._val_step = 0

        self.dummy_input = None
        self.on_gpu = bool('cuda' in device.type)

        # -- Model
        self.build_model(**kwargs)
        self.reset_params()

        # -- Optimizers
        self.optimizer = self.configure_optimizer()
        self.scheduler_optimizer = self.configure_scheduler_optimizer()
        self.reset_train_stats()
        self.reset_val_stats()

    def __getstate__(self):
        return dict((k, v) for k, v in self.__dict__.items()
                    if is_pickable(getattr(self, k)))

    @property
    def _type(self):
        return self.__class__.__name__

    @property
    def trainer(self):
        return self._trainer

    @property
    def epoch(self):
        return self._epoch

    @property
    def global_step(self):
        return self._global_step

    @property
    def train_step(self):
        return self._train_step

    @property
    def val_step(self):
        return self._val_step

    @property
    def early_stop(self):
        return self._early_stop

    # Helpers
    # --------------------------------------------------------
    def warning_not_implemented(self):
        print("")
        warning_not_implemented(self.console_log, level=2)

    def get_maybe_missing_args(self, key, default=None):
        return get_maybe_missing_args(self.args, key, default)

    def set_trainer(self, trainer):
        self._trainer = trainer

    # --------------------------------------------------------

    def build_model(self):
        self._build_model()

    def configure_optimizer(self) -> dict:
        return self._configure_optimizer()

    def configure_scheduler_optimizer(self) -> dict:
        return self._configure_scheduler_optimizer()

    def custom_schedulers(self, *args, **kwargs) -> None:
        self._custom_schedulers(*args, **kwargs)

    def reset_params(self) -> None:
        self._reset_params()

    def training_step(self, batch, epoch, *args, **kwargs) -> dict:
        self._epoch = epoch
        outputs = self._training_step(batch, epoch, *args, **kwargs)
        assert is_dict(outputs), "Output of _training_step should be a dict"
        # Hopefully avoid any memory leak on gpu
        outputs = detach_dict(outputs)
        outputs = to_device(outputs, 'cpu')
        self._train_step += 1
        self._global_step += 1
        return outputs

    def validation_step(self, *args, **kwargs) -> dict:
        outputs = self._validation_step(*args, **kwargs)
        assert is_dict(outputs), "Output of _validation_step should be a dict"
        # Hopefully avoid any memory leak on gpu
        outputs = detach_dict(outputs)
        outputs = to_device(outputs, 'cpu')
        self._val_step += 1
        return outputs

    def test_step(self, *args, **kwargs) -> dict:
        outputs = self._test_step(*args, **kwargs)
        assert is_dict(outputs), "Output of _test_step should be a dict"
        # Hopefully avoid any memory leak on gpu
        outputs = detach_dict(outputs)
        outputs = to_device(outputs, 'cpu')
        return outputs

    # --------------------------------------------------------

    def init_val_stats(self) -> None:
        self.reset_val_stats()

    def init_train_stats(self) -> None:
        self.reset_train_stats()

    def reset_train_stats(self) -> None:
        self._train_step = 0
        self._train_meters = OrderedDict()
        self._reset_train_stats()

    def reset_val_stats(self) -> None:
        self._val_step = 0
        self._val_meters = OrderedDict()
        self._reset_val_stats()

    def log_grads(self):
        norm_type = self.args.loggers.log_grads_norm
        if not norm_type or norm_type < 0:
            return

        if not is_list(norm_type):
            norm_type = [norm_type]

        for _norm in norm_type:
            _grads = self.grad_norm(_norm)
            self.logger.log_metrics(_grads, self._global_step)

    def log_train(self, stats: dict) -> None:
        stats = add_key_dict_prefix(stats, prefix='train', sep='/')
        # Filter out non-scalar items
        metrics = {k: v for k, v in stats.items() if is_scalar(v)}
        self.logger.log_metrics(metrics, self._global_step)
        # self._log_train()

    def log_val(self, descr: str, stats: dict) -> None:
        stats = add_key_dict_prefix(stats, prefix=descr, sep='/')
        # Filter out non-scalar items
        metrics = {k: v for k, v in stats.items() if is_scalar(v)}
        self.logger.log_metrics(metrics, self._epoch)
        # self._log_val()

    # ------------------------------------------------------------------

    @abstractmethod
    def _build_model(self) -> None:
        """
        This method should be implemented specifically for your model.
        Create all your pytorch modules here. This is where all the network
        parameters should be instantiated.
        """
        pass

    def _configure_optimizer(self, parameters=None):
        # -- if not specified, get tall the model parameters
        parameters = self.parameters() if parameters is None else parameters

        args = self.args
        opt_params = self.args.optimizer.params
        weight_decay = self.args.optimizer.regularizers.weight_decay

        # -- Get optimizer from args
        opt_class = get_optimizer(args.optimizer.name)

        # -- Instantiate optimizer with specific parameters
        optimizer = opt_class(
            parameters, weight_decay=weight_decay, **opt_params)
        return optimizer

    def _configure_scheduler_optimizer(self) -> dict:
        args = self.args

        args_scheduler = get_maybe_missing_args(args.optimizer, 'scheduler')
        if args_scheduler is None or not args_scheduler:
            return {}

        scheduler_name = args_scheduler.name
        scheduler_params = args_scheduler.params

        # -- Get optimizer from args
        scheduler_class = get_scheduler_optimizer(scheduler_name)

        # -- Instantiate optimizer with specific parameters
        scheduler = scheduler_class(
            optimizer=self.optimizer, **scheduler_params)
        return scheduler

    # ------------------------------------------------------------------

    @abstractmethod
    def _training_step(self, batch, epoch) -> dict:
        """This method should be implemented specifically for your model.
        It should includes calls to forward and backward of your model
        and to the optimizers. For a concrete example please follow the
        MNIST example implementation.

        Example
        -------
        .. code-block:: python
            def _training_step(self, batch, epoch):
                x, y = batch
                # forward pass
                pred = self.network(x)
                # backward pass
                loss = self.loss_fn(pred, y)
                loss.backward()
                # optimization step
                self.zero_grad()
                self.optimizer.step()
                ...
                # logging statistics
                output = {
                    'running_tqdm': running_tqdm,
                    'final_tqdm': final_tqdm,
                    'stats': stats
                }
                return output
        """
        pass

    def _validation_step(self, *args, **kwargs) -> dict:
        if self.val_step == 0:
            self.warning_not_implemented()
        return {}

    @call_counter
    def _test_step(self, *args, **kwargs) -> dict:
        if self._test_step.calls < 1:
            self.warning_not_implemented()
        return {}

    def _custom_schedulers(self, *args, **kwargs) -> None:
        if self.global_step < 1:
            self.warning_not_implemented()

    def _reset_val_stats(self) -> None:
        self.warning_not_implemented()

    def _reset_train_stats(self) -> None:
        self.warning_not_implemented()

    def _reset_params(self) -> None:
        self.warning_not_implemented()

    # @call_counter
    # def _log_train(self) -> None:
    #     if self._log_train.calls < 1:
    #         self.warning_not_implemented()

    # @call_counter
    # def _log_val(self) -> None:
    #     if self._log_val.calls < 1:
    #         self.warning_not_implemented()

    def early_stopping(self, current_stats: dict) -> (bool, bool):
        """
        This function implements a classic early stopping
        procedure with patience. An example of the arguments that
        can be used is provided.

        ```yml
        early_stopping:
          dataset: 'validation'
          metric: 'validation/y_acc'
          patience: 10
          mode: 'max'              # or 'min'
          train_until_end: False   # it keeps going until the end of training
          warmup: -1               # number of epochs to skip early stopping
                                   # (it disable patience count)

        ```

        Args:
            current_stats (dict): a possibly nested dictionary of the results
                from validation at current epoch. Keys should follow an
                hierarchy as dataset->stats->metric. For custom stats,
                `metric` can be retrieved overriding the method
                `self._get_metric_early_stopping`.

        Returns:
            A tuple (is_best, is_stop) describing the status of early stopping.
            `is_stop` is also assigned to the self object.
        """

        args = get_maybe_missing_args(self.args, 'early_stopping')
        if args is None:
            # -- Do not save in best, and do not stop
            return False, False

        dataset = args.dataset
        metric = args.metric
        patience = args.patience
        mode = args.mode
        train_until_end = get_maybe_missing_args(args, 'train_until_end', False)
        warmup = get_maybe_missing_args(args, 'warmup', -1)
        compare_op = max if mode == "max" else min

        is_best = False
        current = self._get_metric_early_stopping(current_stats)

        if current is None:
            raise ValueError(
                "Metric {} does not exist in current_stats['{}'] \n"
                "It contains only these keys: {}".format(
                    metric, dataset,
                    str(recursive_keys(current_stats))
                )
            )

        # -- first epoch, initialize and do not stop
        if len(self._best_stats) == 0:
            self._best_stats.append((self.epoch, current))
            return True, False

        best = self._best_stats[-1][1]
        if compare_op(current, best) != best:
            self._best_stats.append((self.epoch, current))
            self._best_epoch_score = current
            self._best_epoch = self.epoch
            self._beaten_epochs = 0
            is_best = True
        else:
            if self.epoch > warmup:
                self._beaten_epochs += 1

        if (self._beaten_epochs >= patience and
                not train_until_end and self.epoch > warmup):
            self._early_stop = True

        return is_best, self._early_stop

    def _get_metric_early_stopping(self, current_stats):
        dataset = self.args.early_stopping.dataset
        metric = self.args.early_stopping.metric

        if dataset not in current_stats.keys():
            return None
        if 'stats' not in current_stats[dataset].keys():
            return None

        return current_stats[dataset]['stats'].get(metric, None)

    # --------------------------------------------------------

    # def on_train_start(self):
    #     pass

    # def on_train_end(self):
    #     pass

    # def on_epoch_start(self):
    #     pass

    # def on_epoch_end(self):
    #     pass

    def on_validation_start(self, descr: str) -> None:
        pass

    def on_validation_end(self, descr: str, outputs_list: list = None) -> None:
        pass

    # --------------------------------------------------------

    def freeze(self):
        r"""
        Freeze all params for inference, saving the gradient state
        Example
        -------
        .. code-block:: python
            model = Model(...)
            model.freeze()
        """
        self.requires_grad_snaphot = dict()
        for name, param in self.named_parameters():
            self.requires_grad_snaphot[name] = param.requires_grad
            param.requires_grad = False
        self.eval()

    def unfreeze(self):
        """Unfreeze all params restoring the gradient state before freeze.
        .. code-block:: python
            model = Model(...)
            model.unfreeze()
        """

        for name, param in self.named_parameters():
            param.requires_grad = self.requires_grad_snaphot[name]
        self.train()

    def grad_norm(self, norm_type, sep='/'):
        """
        Module to describe gradients
        """
        results = {}
        total_norm = 0
        for name, p in self.named_parameters():
            if p.requires_grad:
                try:
                    param_norm = p.grad.data.norm(norm_type)
                    total_norm += param_norm ** norm_type
                    norm = param_norm ** (1 / norm_type)

                    grad = round(norm.data.cpu().numpy().flatten()[0], 3)
                    key = 'grads_norm_{}{}{}'.format(norm_type, sep, name)
                    results[key] = grad
                except Exception:
                    # this param had no grad
                    pass

        total_norm = total_norm ** (1. / norm_type)
        grad = round(total_norm.data.cpu().numpy().flatten()[0], 3)
        results['grad_{}_norm_total'.format(norm_type)] = grad
        return results
