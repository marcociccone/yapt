import torch
import logging
import collections
import numpy as np

from ray.tune.schedulers.trial_scheduler import FIFOScheduler, TrialScheduler

from yapt import Trainer
from yapt.utils.utils import flatten_dict
from yapt.core.model.base import BaseModel
from collections import OrderedDict
from abc import abstractmethod
from ray import tune

logger = logging.getLogger(__name__)


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
        self.epoch = self._runner.epoch
        self.extra_args = self._runner.extra_args

        # TODO: make it optioonal with logging
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

        # -- Validation
        val_outputs_flat = OrderedDict()
        for key_loader, val_loader in self._runner.val_loader.items():

            # -- Validate over all datasets
            outputs = self._runner.validate(
                val_loader, log_descr=key_loader, logger=self._runner.logger)

            # - collect and return flatten metrics
            for key_stats, val_stats in outputs['stats'].items():
                if 'scalar' in val_stats.keys():
                    out_flat = flatten_dict(val_stats['scalar'], key_loader)
                elif 'scalars' in val_stats.keys():
                    out_flat = flatten_dict(val_stats['scalars'], key_loader)
                else:
                    out_flat = flatten_dict(val_stats, parent_key=key_loader)
                val_outputs_flat.update(out_flat)

        # -- Be sure that values are scalar and not tensor
        for key, val in val_outputs_flat.items():
            val_outputs_flat[key] = self._get_scalar(val)

        return val_outputs_flat

    def _get_scalar(self, val):
        if isinstance(val, torch.Tensor):
            return val.item()
        return val

    def _save(self, checkpoint_dir):
        return self._runner.save_checkpoint(
            checkpoint_dir, "epoch%d.ckpt" % self._runner.epoch)

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
        self._results = collections.defaultdict(list)
        self._beaten = collections.defaultdict(int)

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

        logger.debug("\nEarlyStoppingRule trial {}: patience {} beaten {} result {} best {}".format(
            trial, self._patience, self._beaten[trial], result[self._metric], best_result))

        if self._beaten[trial] > self._patience:
            logger.debug("\nEarlyStoppingRule: early stopping {}".format(trial))
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
