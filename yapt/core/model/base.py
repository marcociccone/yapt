import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from abc import ABC, abstractmethod


class BaseModel(ABC, nn.Module):

    """Docstring for MyClass. """

    def __init__(self, args, logger=None, **kwargs):
        super().__init__(**kwargs)

        self.args = args
        self.logger = logger
        self.steps = 0

        # -- Model
        self.build_model(**kwargs)
        self.reset_params()

        # -- Optimizers
        self.optimizer = self.configure_optimizer()
        self.scheduler_optimizer = self.configure_scheduler_optimizer()

    @abstractmethod
    def build_model(self) -> None:
        pass

    @abstractmethod
    def training_step(self) -> dict:
        pass

    @abstractmethod
    def configure_optimizer(self) -> dict:
        pass

    def configure_scheduler_optimizer(self) -> dict:
        return {}

    def call_custom_schedulers(self, *args, **kwargs) -> None:
        pass

    def reset_params(self) -> None:
        pass

    def validation_step(self) -> dict:
        pass

    def init_val_stats(self) -> None:
        self.reset_val_stats()

    def reset_val_stats(self) -> None:
        self.meters_val = dict()
        self.running_val_batches = 0

    def init_train_stats(self) -> None:
        self.reset_train_stats()

    def reset_train_stats(self) -> None:
        self.meters = dict()
        self.running_batches = 0

    def log_train(self, stats: dict, logger: SummaryWriter) -> None:
        # Logging on Tensorboard
        for key, val in stats.items():
            logger.add_scalar(
                "train/{}".format(key), val, global_step=self.steps)

    def log_val(self, epoch: int, descr: str, stats: dict, logger: SummaryWriter) -> None:
        # Logging on Tensorboard
        for key, val in stats.items():
            logger.add_scalar(
                "{}/{}".format(descr, key), val, global_step=epoch)

    def early_stopping(self, current_stats: dict, best_stats: dict) -> bool:
        return True, 9999
