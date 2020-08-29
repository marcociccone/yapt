import logging
import torch.nn as nn

from abc import ABC, abstractmethod
from collections import OrderedDict

from yapt import _logger
from yapt import BaseTrainer
from yapt.loggers.base import LoggerDict
from yapt.utils.torch_helpers import (get_optimizer, get_scheduler_optimizer,
                                      detach_dict, to_device)
from yapt.utils.debugging import call_counter, native, is_native
from yapt.utils.args import get_maybe_missing_args
from yapt.utils.utils import (warning_not_implemented,
                              add_key_dict_prefix, is_list, is_scalar,
                              is_dict, recursive_keys)


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
        # Handle the case in which the trainer is not yet been set
        epoch = 0
        if hasattr(self, '_trainer') and self._trainer is not None:
            epoch = self._trainer._epoch
        return epoch

    @property
    def global_step(self):
        # Handle the case in which the trainer is not yet been set
        global_step = 0
        if hasattr(self, '_trainer') and self._trainer is not None:
            global_step = self._trainer.global_step
        return global_step

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
    @native
    def warning_not_implemented(self):
        print("")
        warning_not_implemented(self.console_log, level=2)

    @native
    def get_maybe_missing_args(self, key, default=None):
        return get_maybe_missing_args(self.args, key, default)

    @native
    def set_trainer(self, trainer):
        self._trainer = trainer

    @native
    def check_native(self):
        msg_private = ("{} has been overridden! You should override "
                       "the private method ({}) instead, or decorate "
                       "the function with @native if you really know "
                       "what you are doing!")
        msg = ("{} has been overridden! Decorate the function with @native "
               "if you really know what you are doing!")

        # Retrieve BaseModel's native functions
        native_fn = [name
                     for name, fn in BaseModel.__dict__.items()
                     if is_native(fn)]

        # Check if these functions are still native in self
        for fname in native_fn:
            if not is_native(getattr(self, fname)):
                self.console_log.warning(
                    (msg_private.format(fname, "_"+fname))
                    if hasattr(self, "_"+fname)
                    else msg.format(fname))

    # --------------------------------------------------------

    @native
    def build_model(self):
        self._build_model()

    @native
    def configure_optimizer(self) -> dict:
        return self._configure_optimizer()

    @native
    def configure_scheduler_optimizer(self) -> dict:
        return self._configure_scheduler_optimizer()

    @native
    def custom_schedulers(self, *args, **kwargs) -> None:
        self._custom_schedulers(*args, **kwargs)

    @native
    def reset_params(self) -> None:
        self._reset_params()

    @native
    def training_step(self, batch, *args, **kwargs) -> dict:
        self.train()
        outputs = self._training_step(batch, *args, **kwargs)
        assert is_dict(outputs), "Output of _training_step should be a dict"
        # Hopefully avoid any memory leak on gpu
        outputs = detach_dict(outputs)
        outputs = to_device(outputs, 'cpu')
        self._train_step += 1
        return outputs

    @native
    def validation_step(self, *args, **kwargs) -> dict:
        self.eval()
        outputs = self._validation_step(*args, **kwargs)
        assert is_dict(outputs), "Output of _validation_step should be a dict"
        # Hopefully avoid any memory leak on gpu
        outputs = detach_dict(outputs)
        outputs = to_device(outputs, 'cpu')
        self._val_step += 1
        return outputs

    @native
    def test_step(self, *args, **kwargs) -> dict:
        self.eval()
        outputs = self._test_step(*args, **kwargs)
        assert is_dict(outputs), "Output of _test_step should be a dict"
        # Hopefully avoid any memory leak on gpu
        outputs = detach_dict(outputs)
        outputs = to_device(outputs, 'cpu')
        return outputs

    # --------------------------------------------------------

    @native
    def init_val_stats(self) -> None:
        self.reset_val_stats()

    @native
    def init_train_stats(self) -> None:
        self.reset_train_stats()

    @native
    def reset_train_stats(self) -> None:
        self._train_step = 0
        self._train_meters = OrderedDict()
        self._reset_train_stats()

    @native
    def reset_val_stats(self) -> None:
        self._val_step = 0
        self._val_meters = OrderedDict()
        self._reset_val_stats()

    @native
    def log_grads(self):
        norm_type = self.args.loggers.log_grads_norm
        if not norm_type or norm_type < 0:
            return

        if not is_list(norm_type):
            norm_type = [norm_type]

        for _norm in norm_type:
            _grads = self.grad_norm(_norm)
            self.logger.log_metrics(_grads, self.global_step)

    @native
    def log_train(self, stats: dict) -> None:
        stats = add_key_dict_prefix(stats, prefix='train', sep='/')
        # Filter out non-scalar items
        metrics = {k: v for k, v in stats.items() if is_scalar(v)}
        self.logger.log_metrics(metrics, self.global_step)
        # self._log_train()

    @native
    def log_val(self, descr: str, stats: dict) -> None:
        stats = add_key_dict_prefix(stats, prefix=descr, sep='/')
        # Filter out non-scalar items
        metrics = {k: v for k, v in stats.items() if is_scalar(v)}
        self.logger.log_metrics(metrics, self.epoch)
        # self._log_val()

    @native
    def aggregate_accum_stats(self, accum_stats_list: list) -> dict:
        """
        This function is called to aggregate statistics from consecutive
        iterations of gradients accumulation before logging the statistics
        on Neptune or Tensorboard. By default the mean of each key from
        consecutive runs is returned. Please override `_aggregate_accum_stats`
        instead on this function, if you want to change its behavior.

        Args:
            accum_stats_list (List[Dict]): A list of dictionaries of length
                accum_batches, each one containing the output['stats'] dict
                returned by the corresponding _training_step() method.

        Returns:
            A dictionary containing aggregate statistics from multiple
            accumulation steps.

        """
        # Skip if there is only one step to aggregate
        # (i.e., accum_batches = 1)
        if len(accum_stats_list) == 1:
            return accum_stats_list[0]

        # Convert from list of dicts to dict of lists
        paired_stats = {}
        for stats in accum_stats_list:
            for key, val in stats.items():
                if key not in paired_stats:
                    paired_stats[key] = []
                paired_stats[key].append(val)

        # Call the user overridable function
        return self._aggregate_accum_stats(paired_stats)

    def _aggregate_accum_stats(self, paired_stats: dict):
        """
        This function is called to aggregate statistics from consecutive
        iterations of gradients accumulation before logging the statistics
        on Neptune or Tensorboard. By default the mean of each key from
        consecutive runs is returned, i.e.,
        .. code-block:: python
            return {k: mean(v) for k, v in paired_stats.items()}

        Args:
            paired_stats (Dict[str, List]): A dictionary of lists containing the
            same keys of dictionaries returned by _training_step(), but in which
            values are replaced by lists of length accum_batches. Each list
            collects the sequence of values of the statistics in consecutive
            grads accumulation iterations.

        Returns:
            A dictionary containing aggregate statistics from multiple
            accumulation steps.
        """
        try:
            # Aggregate with mean
            return {k: sum(v) / len(v) for k, v in paired_stats.items()}
        except:
            self.console_log.exception(
                "model._aggregate_accum_stats() was not able to aggregate "
                "statistics from consecutive gradient accumulation steps. "
                "Please implement a custom _aggregate_accum_stats().")

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
    def _training_step(self, batch) -> dict:
        """This method should be implemented specifically for your model.
        It should includes calls to forward and backward of your model
        and to the optimizers. For a concrete example please follow the
        MNIST example implementation.

        Example
        -------
        .. code-block:: python
            def _training_step(self, batch):
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

    @call_counter
    def _custom_schedulers(self, *args, **kwargs) -> None:
        if self._custom_schedulers.calls < 1:
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

    def _update_best_stats(self, current):
        """Update statistics and variables regarding best model.
        """
        self._best_stats.append((self.epoch, current))
        self._best_epoch_score = current
        self._best_epoch = self.epoch
        self._beaten_epochs = 0

    @native
    def early_stopping(self, current_stats: dict) -> (bool, bool):
        """This function implements a classic early stopping
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
            self._update_best_stats(current)
            return True, False

        best = self._best_stats[-1][1]
        if compare_op(current, best) != best:
            self._update_best_stats(current)
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

    @native
    def _init_optimizer_accum(self):
        """
        Initialize a dictionary of counters to keep track, for each optimizer,
        how many times the `optimizer.step()` function has been called. The
        optimizer itself is used as key of the dictionary.
        """
        # If not already, convert into a dict for convenience
        optim_dict = (self.optimizer
                      if is_dict(self.optimizer)
                      else {'optimizer': self.optimizer})
        # -- Initialize the dict of steps per optimizer
        self._optimizers_accum = {key: 0 for key in optim_dict.values()}

        # -- Decorate optimizer's step() and zero_grad() with counters
        for optim in optim_dict.values():
            optim.step = call_counter(optim.step)
            optim.zero_grad = call_counter(optim.zero_grad)

        # -- Decorate model zero_grad
        self.zero_grad = call_counter(self.zero_grad)

    @native
    def _check_optimizer_accum(self):
        """
        Check if helper functions for `loss.backward()`, `optim.step()` and
        `optim.zero_grad()` are used instead of the pytorch originals when
        gradient accumulation or clipping is used.
        """
        # If not already, convert into a dict for convenience
        optim_dict = (self.optimizer
                      if is_dict(self.optimizer)
                      else {'optimizer': self.optimizer})

        clip_val = self.args.grads_norm_clip.max_norm
        accum_batches = self.args.accum_batches
        if clip_val > 0 or accum_batches > 1:
            # -- Warning messages stuff
            warn_pattern = ("{1} has been called directly while using {0}, "
                            "which will prevent it from working. Please use "
                            "{2} instead.")
            used_methods = {'grads clipping': clip_val > 0,
                            'grads accumulation': accum_batches > 1}
            methods = "".join(["%s and " % name
                               for name, active in used_methods.items()
                               if active])[:-len(" and ")]

            # -- Check if optimizer.step() has been directly called
            steps_calls = [o.step.calls for o in optim_dict.values()]
            if max(steps_calls) > 0:
                self.console_log.warning(
                    warn_pattern.format(methods, "optimizer.step()",
                                        "self.optimize_step(optimizer)"))

            # -- Check if optimizer.zero_grad() has been directly called
            zero_calls = [o.zero_grad.calls for o in optim_dict.values()]
            if max(zero_calls) > 0:
                self.console_log.warning(
                    warn_pattern.format(methods, "optimizer.zero_grad()",
                                        "self.zero_grad_step(optimizer)"))

            # -- Check if model.zero_grad() has been directly called
            if self.zero_grad.calls > 0:
                self.console_log.warning(
                    warn_pattern.format(methods, "self.zero_grad()",
                                        "self.zero_grad_step(optimizer)"))

            # -- Check if self.compute_grad_step has been called at least once
            if self.compute_grad_step.calls == 0:
                self.console_log.warning(
                    warn_pattern.format(methods, "loss.backward()",
                                        "self.compute_grad_step(loss)"))

    @native
    @call_counter
    def compute_grad_step(self, loss):
        """
        Compute the backpropagation step given a loss. This function optionally
        scales the loss by the number of gradient accumulation steps before
        backward if grads accumulation (accum_batches > 1) is enabled. Use this
        function in place of `loss.backward()` if you want to use YAPT's
        gradient accumulation.

        Args:
            loss (torch.Tensor): The pytorch leaf tensor to backpropagate
                gradient from
        """
        # -- Scale the loss by the number of accumulation steps
        scaled_loss = loss / self.args.accum_batches
        scaled_loss.backward()

    @native
    def zero_grad_step(self, optimizer):
        """
        Zero the parameter gradients of the parameters controlled by the given
        optimizer. If grads accumulation is enabled (accum_batches > 1),
        gradients zeroing is only performed once every `accum_batches` steps of
        the current optimizer. Use this function in place of `optim.zero_grad()`
        if you want to use YAPT's gradient accumulation.

        Args:
            optimizer (torch.optim.Optimizer): The optimizer handling the
                parameters to be zeroed
        """
        # -- Zero the parameter gradients only
        # after accum_batches calls of step()
        steps = self._optimizers_accum[optimizer]
        if steps % self.args.accum_batches == 0:
            # -- Zero the parameters gradients
            optimizer.zero_grad()
            optimizer.zero_grad.calls -= 1

    @native
    def optimize_step(self, optimizer):
        """
        Perform the optimization step, updating the model's parameters based
        on their gradients. If grads accumulation is enabled (accum_batches > 1)
        optimization is only performed once every `accum_batches`. This function
        also optionally performs gradient clipping on the parameters handled by
        the given optimizer, if required using `grads_norm_clip` arguments. Use
        this function in place of `optim.step()` if you want to use YAPT's
        gradient accumulation and clipping.

        Args:
            optimizer (torch.optim.Optimizer): The optimizer handling the
                parameters to be updated
        """
        # -- Increase number of steps done by the optimizer
        self._optimizers_accum[optimizer] += 1
        steps = self._optimizers_accum[optimizer]

        # -- Perform the step only if step(optimizer)
        # has been called accum_batches times
        if steps % self.args.accum_batches == 0:
            # -- Apply grads clipping
            clip_val = self.args.grads_norm_clip.max_norm
            norm_type = self.args.grads_norm_clip.norm_type
            if clip_val is not None and clip_val > 0:
                # Concatenate all parameter groups parameters, i.e., all
                # parameters handled by the current optimizer
                optim_params = sum([pg['params']
                                    for pg in optimizer.param_groups], [])
                # Clip only the parameters of the current optimizer
                nn.utils.clip_grad_norm_(optim_params,
                                         max_norm=clip_val,
                                         norm_type=norm_type)

            # -- Perform the actual step
            optimizer.step()
            optimizer.step.calls -= 1

    @native
    def update_step(self, optimizer, loss):
        """
        The update step, i.e., gradients zeroing, gradient computation and
        parameter update. This is an helper function that aggregates the three
        steps into a single function. Use this function is you want to use
        YAPT's gradient accumulation and clipping, or the three separate
        `zero_grad_step()`, `compute_grad_step()` and `optimize_step()` if you
        want to adopt a different update logic for whatever reason.

        Args:
            optimizer (torch.optim.Optimizer): The optimizer handling the
                parameters to be updated
            loss (torch.Tensor): The pytorch leaf tensor to backpropagate
                gradient from
        """
        self.zero_grad_step(optimizer)
        self.compute_grad_step(loss)
        self.optimize_step(optimizer)

    # --------------------------------------------------------

    def _on_train_start(self):
        pass

    def _on_train_end(self):
        pass

    def _on_epoch_start(self):
        pass

    def _on_epoch_end(self):
        pass

    def _on_validation_start(self, descr: str) -> None:
        pass

    def _on_validation_end(self, descr: str, outputs_list: list = None) -> None:
        pass

    @native
    def on_train_start(self):
        # -- Setup counters for grads accum
        self._init_optimizer_accum()

        # -- Check native functions
        self.check_native()

        return self._on_epoch_start()

    @native
    def on_train_end(self):
        return self._on_train_end()

    @native
    def on_epoch_start(self):
        return self._on_epoch_start()

    @native
    def on_epoch_end(self):
        # -- Check grads accumulation
        self._check_optimizer_accum()
        return self._on_epoch_end()

    @native
    def on_validation_start(self, descr: str) -> None:
        return self._on_validation_start(descr)

    @native
    def on_validation_end(self, descr: str, outputs_list: list = None) -> None:
        return self._on_validation_end(descr, outputs_list)

    # --------------------------------------------------------

    @native
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

    @native
    def unfreeze(self):
        """Unfreeze all params restoring the gradient state before freeze.
        .. code-block:: python
            model = Model(...)
            model.unfreeze()
        """

        for name, param in self.named_parameters():
            param.requires_grad = self.requires_grad_snaphot[name]
        self.train()

    @native
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
