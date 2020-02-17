import argparse
import ray

from ray import tune
from ray.tune.schedulers import ASHAScheduler

from yapt import TuneWrapper, EarlyStoppingRule
from mnist_trainer import TrainerMNIST
from model import Classifier

import logging
logging.basicConfig(level=logging.DEBUG)


class TuneMNIST(TuneWrapper):

    def _setup(self, config):
        self.trainer = TrainerMNIST(extra_args=config, model_class=Classifier)
        super()._setup(config)


if __name__ == "__main__":

    # Training settings
    parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
    parser.add_argument(
        "--use-gpu", action="store_true", default=False, help="CUDA training")
    parser.add_argument(
        "--ray-address", type=str, help="The Redis address of the cluster.")
    parser.add_argument(
        "--smoke-test", action="store_true", help="Finish quickly for testing")

    args = parser.parse_args()

    tune_config = {
        'general': {'dry_run': False},
        'tqdm': {'disable': True},
        'optimizer': {
            'name': 'sgd',
            'params': {
                'lr': tune.uniform(0.001, 0.1),
                'momentum': tune.uniform(0.1, 0.9)
            }
        }
    }

    # -- Ray initialization and scheduler
    ray.init(address=args.ray_address, log_to_driver=True)
    # sched = ASHAScheduler(metric="acc")
    sched = EarlyStoppingRule(metric="acc", patience=10)

    analysis = tune.run(
        TuneMNIST,
        scheduler=sched,
        stop={
            "training_iteration": 150
        },
        resources_per_trial={
            "cpu": 3,
            "gpu": 0.1
        },
        num_samples=10,
        checkpoint_at_end=True,
        checkpoint_freq=10,
        config=tune_config,
        local_dir='./logs'
    )

    print("Best config is:", analysis.get_best_config(metric="acc"))

