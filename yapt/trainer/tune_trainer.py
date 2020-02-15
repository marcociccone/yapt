from abc import abstractmethod
from ray import tune


class TuneWrapper(tune.Trainable):

    @abstractmethod
    def _setup(self, config):
        self.trainer = None
        self.model = self.trainer.model
        self.epoch = self.trainer.epoch

    def _train(self):
        args = self.trainer.args

        if args.trainer.dry_run:
            self.trainer.extra_args.pretty()
            self.stop()
            return

        # -- Training epoch
        self.trainer.train_epoch(self.trainer.train_loader['labelled'])
        self.epoch = self.trainer.epoch

        # -- Validation
        outputs = self.trainer.validate(
            self.trainer.val_loader['validation'],
            log_descr='validation',
            logger=self.trainer.logger)

        return outputs['stats']

    def _save(self, checkpoint_dir):
        return self.trainer.save_checkpoint(
            checkpoint_dir, "epoch%d.ckpt" % self.epoch)

    def _restore(self, checkpoint_path):
        self.trainer.load_checkpoint(checkpoint_path)
