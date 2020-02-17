import torch
import logging
import collections
import numpy as np

from ray.tune.trial import Trial
from ray.tune.schedulers.trial_scheduler import FIFOScheduler, TrialScheduler

from abc import abstractmethod
from ray import tune

logger = logging.getLogger(__name__)


class TuneWrapper(tune.Trainable):

    @abstractmethod
    def _setup(self, config):
        self.args = self.trainer.args
        self.model = self.trainer.model
        self.epoch = self.trainer.epoch
        self.extra_args = self.trainer.extra_args
        print(self.args.pretty())
        print(self.extra_args.pretty())

    def _train(self):
        args = self.args
        if args.dry_run:
            print(args.extra_args.pretty())
            self.stop()
            return {}

        # -- Training epoch
        self.trainer.train_epoch(self.trainer.train_loader['labelled'])
        self.epoch = self.trainer.epoch

        # -- Validation
        val_set = args.early_stopping.get('dataset', 'validation')
        outputs = self.trainer.validate(
            self.trainer.val_loader[val_set],
            log_descr=val_set,
            logger=self.trainer.logger)

        for key, val in outputs['stats'].items():
            if isinstance(val, torch.Tensor):
                outputs['stats'][key] = val.item()
        return outputs['stats']

    def _save(self, checkpoint_dir):
        return self.trainer.save_checkpoint(
            checkpoint_dir, "epoch%d.ckpt" % self.trainer.epoch)

    def _restore(self, checkpoint_path):
        self.trainer.load_checkpoint(checkpoint_path)


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
        This stopping rule stops a running trial if the trial's best objective
        value by step `t` is strictly worse than the median of the running
        averages of all completed trials' objectives reported up to step `t`.
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
