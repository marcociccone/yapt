import torch
import logging

from collections import OrderedDict, defaultdict
from abc import abstractmethod

from yapt import Trainer
from yapt.utils.utils import flatten_dict, is_dict
from yapt.core.model import BaseModel
from yapt import _logger

from ray import tune
from ray.tune.schedulers.trial_scheduler import FIFOScheduler, TrialScheduler


class TuneWrapper(tune.Trainable):

    @abstractmethod
    def _build_runner(self, extra_args, logdir):
        """
            Template: replace with your trainer and model
        """
        return Trainer(extra_args=extra_args,
                       external_logdir=logdir,
                       model_class=BaseModel)

    def _setup(self, config):

        self._runner = self._build_runner(config, self._result_logger.logdir)
        self.args = self._runner.args
        self.model = self._runner.model
        self.extra_args = self._runner.extra_args

        # TODO: make it optional with logging
        print(self.args.pretty())
        print(self.extra_args.pretty())

    def _train(self):
        args = self.args
        if args.dry_run:
            print(args.extra_args.pretty())
            self.stop()
            return {}

        # -- Training epoch
        # TODO: now labelled is hardcoded, make it general
        self._runner.train_epoch(self._runner.train_loader['labelled'])
        self.epoch = self._runner.epoch

        # -- Validate over all datasets
        val_outputs_flat = OrderedDict()
        for key_loader, val_loader in self._runner.val_loader.items():
            num_batches = self._runner.num_batches_val[key_loader]
            # -- Validation on val_loader
            outputs = self._runner.validate(
                val_loader, num_batches=num_batches,
                set_name=key_loader, logger=self._runner.logger)

            # -- TODO: flatten_Dict should not be necessary,
            # -- prefix key is already concatenated in validate method
            # -- collect and return flatten metrics
            for key_stats, val_stats in outputs['stats'].items():
                if is_dict(val_stats):
                    if 'scalar' in val_stats.keys():
                        _flat = flatten_dict(val_stats['scalar'], False)
                    elif 'scalars' in val_stats.keys():
                        _flat = flatten_dict(val_stats['scalars'], False)
                    else:
                        _flat = flatten_dict(val_stats, False)
                else:
                    _flat = {key_stats: val_stats}
                val_outputs_flat.update(_flat)

        # -- Be sure that values are scalar and not tensor
        remove_keys = []
        for key, val in val_outputs_flat.items():
            if val.dim() == 0:
                val_outputs_flat[key] = self._get_scalar(val)
            else:
                remove_keys.append(key)

        for key in remove_keys:
            del val_outputs_flat[key]

        return val_outputs_flat

    def _get_scalar(self, val):
        if isinstance(val, torch.Tensor):
            return val.item()
        return val

    def _save(self, checkpoint_dir):
        return self._runner.save_checkpoint(path=checkpoint_dir)

    def _restore(self, checkpoint_path):
        # TODO: this has to be checked
        self._runner.load_checkpoint(checkpoint_path)


class EarlyStoppingRule(FIFOScheduler):
    """Implements a simple early-stopping rule with patience
    Args:
        metric (str): The training result objective value attribute. Stopping
            procedures will use this attribute.
        mode (str): One of {min, max}. Determines whether objective is
            minimizing or maximizing the metric attribute.
        hard_stop (bool): If False, pauses trials instead of stopping
            them. When all other trials are complete, paused trials will be
            resumed and allowed to run FIFO.
        patience (int): Number of times the metric can be worse than the best
    """

    def __init__(self,
                 metric="acc",
                 mode='max',
                 patience=1,
                 hard_stop=True):

        FIFOScheduler.__init__(self)
        self._metric = metric
        assert mode in {"min", "max"}, "`mode` must be 'min' or 'max'."
        self._worst = float("-inf") if mode == "max" else float("inf")
        self._compare_op = max if mode == "max" else min
        self._hard_stop = hard_stop
        self._patience = patience
        self._results = defaultdict(list)
        self._beaten = defaultdict(int)

    def on_trial_result(self, trial_runner, trial, result):
        """Callback for early stopping.
        This stopping rule stops a running trial if the trial's objective
        does not improve for 'patience' number of time.
        """

        if len(self._results[trial]) == 0:
            self._results[trial].append(result)
            return TrialScheduler.CONTINUE

        best_result = self._best_result(trial)
        self._results[trial].append(result)

        if self._compare_op(result[self._metric], best_result) != best_result:
            self._beaten[trial] = 0
        else:
            self._beaten[trial] += 1

        _logger.debug("\nEarlyStoppingRule trial {}: patience {} beaten {} result {} best {}".format(
            trial, self._patience, self._beaten[trial], result[self._metric], best_result))

        if self._beaten[trial] > self._patience:
            _logger.debug("\nEarlyStoppingRule: early stopping {}".format(trial))
            if self._hard_stop:
                return TrialScheduler.STOP
            else:
                return TrialScheduler.PAUSE
        else:
            return TrialScheduler.CONTINUE

    def on_trial_complete(self, trial_runner, trial, result):
        self._results[trial].append(result)

    def _best_result(self, trial):
        results = self._results[trial]
        return self._compare_op([r[self._metric] for r in results])
