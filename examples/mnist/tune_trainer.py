import os
import ray

from ray import tune
from ray.tune.schedulers import ASHAScheduler

from yapt import TuneWrapper, EarlyStoppingRule
from mnist_trainer import TrainerMNIST
from model import Classifier


class TuneMNIST(TuneWrapper):

    def _build_runner(self, config, logdir):
        return TrainerMNIST(extra_args=config,
                            external_logdir=logdir,
                            model_class=Classifier)


if __name__ == "__main__":

    tune_config = {
        'dry_run': False,
        'loggers': {'tqdm': {'disable': True}},
        'optimizer': {
            'name': 'sgd',
            'params': {
                'lr': tune.uniform(0.001, 0.1),
                'momentum': tune.uniform(0.1, 0.9)
            }
        }
    }

    # -- Ray initialization and scheduler
    # -- NOTE: local_mode=True for debugging
    ray.init(address=None, log_to_driver=True, local_mode=True)
    # sched = ASHAScheduler(metric="validation/y_acc")
    sched = EarlyStoppingRule(metric="validation/y_acc", patience=10)

    analysis = tune.run(
        TuneMNIST,
        scheduler=sched,
        stop={
            "training_iteration": 25
        },
        resources_per_trial={
            "cpu": 3,
            "gpu": 0.1
        },
        num_samples=1,
        checkpoint_at_end=True,
        checkpoint_freq=1,
        config=tune_config,
        local_dir=os.path.join(
            os.environ['YAPT_LOGDIR'], 'mnist', 'tune_example')
    )

    print("Best config is:",
          analysis.get_best_config(metric="validation/y_acc"))

