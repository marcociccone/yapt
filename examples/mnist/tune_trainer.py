import argparse
import ray

from ray import tune
from ray.tune.schedulers import ASHAScheduler

from mnist_trainer import TrainerMNIST
from model import Classifier


class TuneMNIST(tune.Trainable):
    def _setup(self, config):
        self.trainer = TrainerMNIST(extra_args=config, model_class=Classifier)

    def _train(self):
        # -- Training epoch
        self.trainer.train_epoch(self.trainer.train_loader['labelled'])
        # -- Validation
        outputs = self.trainer.validate(
            self.trainer.val_loader['validation'],
            log_descr='validation',
            logger=self.trainer.logger)

        return {"mean_accuracy": outputs['stats']['acc']}

    def _save(self, checkpoint_dir):
        return self.trainer.save_checkpoint(checkpoint_dir, 'model.pt')

    def _restore(self, checkpoint_path):
        self.trainer.load_checkpoint(checkpoint_path)


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
        'optimizer.name': 'sgd',
        'optimizer.params': {
            "lr": tune.uniform(0.001, 0.1),
            "momentum": tune.uniform(0.1, 0.9)
        }
    }

    # -- Ray initialization and scheduler
    ray.init(address=args.ray_address, log_to_driver=False)
    sched = ASHAScheduler(metric="mean_accuracy")

    analysis = tune.run(
        TuneMNIST,
        scheduler=sched,
        stop={
            "mean_accuracy": 0.99,
            "training_iteration": 50
        },
        resources_per_trial={
            "cpu": 3,
            "gpu": int(args.use_gpu)
        },
        num_samples=1 if args.smoke_test else 20,
        checkpoint_at_end=True,
        checkpoint_freq=3,
        config=tune_config
    )

    print("Best config is:", analysis.get_best_config(metric="mean_accuracy"))

