import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from yapt.utils.utils import call_counter, warning_not_implemented
from abc import ABC, abstractmethod


class BaseModel(ABC, nn.Module):

    """Docstring for MyClass. """

    def __init__(self, args, logger=None, device='cpu', **kwargs):
        super().__init__(**kwargs)

        self.args = args
        self.logger = logger
        self.device = device

        self.epoch = 0
        self.global_step = 0
        self.train_step = 0
        self.val_step = 0

        # -- Model
        self.build_model(**kwargs)
        self.reset_params()

        # -- Optimizers
        self.optimizer = self.configure_optimizer()
        self.scheduler_optimizer = self.configure_scheduler_optimizer()

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
        self.epoch = epoch
        outputs = self._training_step(batch, epoch, *args, **kwargs)
        self.train_step += 1
        self.global_step += 1
        return outputs

    def validation_step(self, *args, **kwargs) -> dict:
        outputs = self._validation_step(*args, **kwargs)
        self.val_step += 1
        return outputs

    def test_step(self, *args, **kwargs) -> dict:
        outputs = self._test_step(*args, **kwargs)
        return outputs

    def init_val_stats(self) -> None:
        self.reset_val_stats()

    def init_train_stats(self) -> None:
        self.reset_train_stats()

    def reset_train_stats(self) -> None:
        self.train_step = 0
        self._reset_train_stats()

    def reset_val_stats(self) -> None:
        self.val_step = 0
        self._reset_val_stats()

    def log_train(self, stats: dict, logger: SummaryWriter) -> None:
        # Logging on Tensorboard
        for key, val in stats.items():
            logger.add_scalar(
                "train/{}".format(key), val, global_step=self.global_step)
        # self._log_train()

    def log_val(self, epoch: int, descr: str, stats: dict, logger: SummaryWriter) -> None:
        # Logging on Tensorboard
        for key, val in stats.items():
            logger.add_scalar(
                "{}/{}".format(descr, key), val, global_step=epoch)
        # self._log_val()

    # ------------------------------------------------------------------
    @abstractmethod
    def _build_model(self) -> None:
        pass

    @abstractmethod
    def _configure_optimizer(self) -> dict:
        pass

    def _configure_scheduler_optimizer(self) -> dict:
        warning_not_implemented()
        return {}

    @abstractmethod
    def _training_step(self, batch, epoch) -> dict:
        pass

    def _validation_step(self, *args, **kwargs) -> dict:
        if self.val_step == 0:
            warning_not_implemented()
        return {}

    @call_counter
    def _test_step(self, *args, **kwargs) -> dict:
        if self._test_step.calls < 1:
            warning_not_implemented()
        return {}

    def _custom_schedulers(self, *args, **kwargs) -> None:
        if self.global_step < 1:
            warning_not_implemented()

    def _reset_val_stats(self) -> None:
        warning_not_implemented()

    def _reset_train_stats(self) -> None:
        warning_not_implemented()

    def _reset_params(self) -> None:
        warning_not_implemented()

    # @call_counter
    # def _log_train(self) -> None:
    #     if self._log_train.calls < 1:
    #         warning_not_implemented()

    # @call_counter
    # def _log_val(self) -> None:
    #     if self._log_val.calls < 1:
    #         warning_not_implemented()

    def early_stopping(self, current_stats: dict, best_stats: dict) -> bool:
        return True, 9999
