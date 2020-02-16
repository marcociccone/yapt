import torch
import numpy as np

from itertools import cycle, islice

from yapt.utils.trainer_utils import alternate_datasets, detach_dict, to_device
from yapt.utils.utils import is_notebook, stats_to_str

from yapt.trainer.sacred_trainer import SacredTrainer

if is_notebook():
    from tqdm import tqdm_notebook as tqdm
else:
    from tqdm import tqdm


class Trainer(SacredTrainer):
    def __init__(self, data_loaders=None, *args, **kwargs):

        super().__init__(*args, **kwargs)
        args = self.args

        # -- 1. Load Datasets
        data_loaders = self.set_data_loaders() \
            if data_loaders is None else data_loaders
        self.init_data_loaders(data_loaders)

        self.seen = 0
        self.epoch = 1
        self.best_epoch = 1
        self.best_epoch_score = 0
        self.beaten_epochs = 0
        self.best_epoch_output_train = dict()
        self.best_epoch_output_val = dict()
        self.outputs_train = list()
        self.outputs_val = list()

        # -- Early Stopping
        self.max_epochs = args.max_epochs
        self.early_stopping = args.early_stopping

    def init_data_loaders(self, data_loaders: dict):

        assert data_loaders.get('train', None) is not None, \
            "train_loader cannot be None"
        self.train_loader = data_loaders['train']
        self.val_loader = data_loaders.get('val', None)
        self.test_loader = data_loaders.get('test', None)

        self.semi_supervised = self.args.trainer.semi_supervised

        if self.semi_supervised:
            assert isinstance(self.train_loader, dict), \
                "train_loader should be a dict()"
            assert self.train_loader.get('labelled', None) is not None, \
                "labelled dataloader is None"
            assert self.train_loader.get('unlabelled', None) is not None, \
                "unlabelled dataloader is None"

            self.num_batches_train = max(
                len(self.train_loader['labelled']),
                len(self.train_loader['unlabelled']))
        else:
            self.num_batches_train = len(self.train_loader)
            if isinstance(self.train_loader, dict):
                assert 'labelled' in self.train_loader.keys(), \
                    "Train loader dict should contain 'labelled' key"
            else:
                self.train_loader = {'labelled': self.train_loader}

        if self.val_loader is not None:
            if not isinstance(self.val_loader, dict):
                self.val_loader = {'validation': self.val_loader}

    def train_epoch(self, dataloader):
        """
        Performs one entire epoch of training
        :param dataloader: A DataLoader object producing training samples
        :return: a tuple (epoch_loss, epoch_accuracy)
        """

        # -- Initialization and training mode

        # Enters train mode
        self.model.train()
        # Zero the parameter gradients
        self.model.zero_grad()
        # Track training statistist
        self.model.init_train_stats()
        self.outputs_train.append([])

        pbar_descr_prefix = "Epoch %d (best: %d, beaten: %d) - " % (
            self.epoch, self.best_epoch, self.beaten_epochs)

        pbar = tqdm(total=self.num_batches_train,
                    desc=pbar_descr_prefix,
                    **self.args.tqdm)

        # -- Start epoch
        for batch_idx, batch in enumerate(dataloader):
            device_batch = self.to_device(batch)

            # -- Model specific schedulers
            self.model.custom_schedulers(
                self.epoch, logger=self.logger)

            # -- Execute a training step
            outputs = self.model.training_step(
                device_batch, self.epoch)

            # -- Save output for each training step
            self.collect_outputs(outputs)

            # -- Eventually log statistics on tensorboard
            if self.model.global_step % self.log_every == 0:
                self.model.log_train(
                    outputs.get('stats', dict()), self.logger)

            running_tqdm = stats_to_str(outputs.get('running_tqdm', dict()))
            pbar.set_description(pbar_descr_prefix + running_tqdm)
            pbar.update()

        pbar.clear()
        pbar.close()

        # -- End Epoch
        self.model.reset_train_stats()
        final_tqdm = self.outputs_train[-1][-1].get('final_tqdm', dict())
        print(pbar_descr_prefix + stats_to_str(final_tqdm))
        return self.outputs_train[-1]

    def _fit(self):
        """
         A complete training procedure by performing early stopping using the provided validation set
        """

        self.seen = 0
        self.epoch = 1
        self.best_epoch = 1
        self.beaten_epochs = 0
        self.best_epoch_score = 0
        self.best_epoch_output_train = dict()
        self.best_epoch_output_val = dict()

        if self.early_stopping is not None:
            self.print_verbose(
                "Early stopping: set {} - metric {} - patience {}".format(
                    self.early_stopping.dataset,
                    self.early_stopping.metric,
                    self.early_stopping.patience))

        # Loads the initial checkpoint if provided
        if self.restart_path is not None:
            self.restart_exp()

        self.log_args()
        self.dump_args(self.logdir)

        while self.epoch < self.max_epochs:

            if (self.early_stopping is not None and
                self.beaten_epochs > self.early_stopping.patience):
                break

            # -- TODO: move out
            if self.semi_supervised:
                if self.args.alternated_update:
                    train_loader = alternate_datasets(
                        iter(self.train_loader['labelled']),
                        iter(self.train_loader['unlabelled']))
                else:
                    train_loader = zip(
                        islice(cycle(self.train_loader['labelled']),
                               self.num_batches_train),
                        self.train_loader['unlabelled'])
            else:
                train_loader = self.train_loader['labelled']

            # -- Call Optimizer schedulers and track lr
            if self.epoch > 1:
                # -- Pytorch 1.1.0 requires to call first optimizer.step()
                self.call_schedulers_optimizers()
            self.log_each_epoch()

            # -- Performs one epoch of training
            output_train = self.train_epoch(train_loader)
            self.save_checkpoint(self.logdir, "epoch%d.ckpt" % self.epoch)

            # -- Validate the network against the validation set
            output_val = dict()
            for kk, vv in self.val_loader.items():
                output_val[kk] = self.validate(
                    vv, log_descr=kk, logger=self.logger)
            print("")

            is_best_epoch, best_score = self.model.early_stopping(
                output_val, self.best_epoch_output_val)

            if is_best_epoch or self.epoch == 1:
                self.best_epoch = self.epoch
                self.best_epoch_score = best_score
                self.best_epoch_output_train = output_train
                self.best_epoch_output_val = output_val
                self.beaten_epochs = 0
            else:
                self.beaten_epochs += 1

            self.epoch += 1

        print("Reloading best epoch %d checkpoint" % self.best_epoch)
        self.load_checkpoint(self.logdir, "epoch%d.ckpt" % self.best_epoch)

        if self.test_loader is not None:
            # TODO !!!
            output_test = self.validate(
                self.test_loader, log_descr="test", logger=None)
            self.json_results(self.logdir, output_test)
        else:
            output_test = output_val[kk]

        return output_test

    def validate(self, dataloader, log_descr="validation", logger=None):
        """
        Computes the accuracy of the network against a validation set
        :param dataloader: A DataLoader object producing validation/test samples
        :return: the accuracy over the validation dataset
        """
        if dataloader is None:
            return {}

        # Enters eval mode
        self.model.eval()
        self.model.init_val_stats()

        pbar_descr_prefix = "\t%s - " % log_descr.title()

        # TODO!!
        # Disable network grad while evaluating the model
        # with no_grad_ifnotscript(self.model):
        if torch.is_grad_enabled():
            print("WARNING: You should handle no_grad by yourself for now!")

        # --------------------------------------------------------------------

        pbar = tqdm(total=len(dataloader), desc=pbar_descr_prefix,
                    **self.args.tqdm)

        for batch_idx, batch in enumerate(dataloader):
            device_batch = self.to_device(batch)
            outputs = self.model.validation_step(device_batch, self.epoch)

            running_tqdm = stats_to_str(outputs.get('running_tqdm', dict()))
            pbar.set_description(pbar_descr_prefix + running_tqdm)
            pbar.update()

        pbar.clear()
        pbar.close()

        # --------------------------------------------------------------------

        if logger is not None:
            self.model.log_val(
                self.epoch, log_descr,
                outputs.get('stats', dict()), logger)

        self.model.reset_val_stats()
        final_tqdm = stats_to_str(outputs.get('final_tqdm', dict()))
        print("\t%s - %s" % (log_descr.title(), final_tqdm))
        return outputs

    def only_test(self):

        # Reload last or best epoch
        if self.restart_path is not None:
            self.restart_exp()
        else:
            raise ValueError("Give me the folder experiment!")

        print("Reloading best epoch %d checkpoint" % self.best_epoch)
        self.load_checkpoint(self.logdir, "epoch%d.ckpt" % self.best_epoch)

        if self.test_loader is not None:
            output_test = self.validate(
                self.test_loader, log_descr="test", logger=None)
            self.json_results(self.logdir, output_test)
            return output_test

    def test(self, dataloader, to_numpy=True):

        # Load the last / best checkpoint
        if self.restart_path is not None:
            self.restart_exp()
        else:
            raise ValueError("Give me the folder experiment!")

        print("Reloading best epoch %d checkpoint" % self.best_epoch)
        self.load_checkpoint(self.logdir, "epoch%d.ckpt" % self.best_epoch)

        out_dict = {}
        if torch.is_grad_enabled():
            print("WARNING: You should handle no_grad by yourself for now!")

        pbar = tqdm(total=len(dataloader), desc='test',
                    **self.args.tqdm)

        for batch_idx, batch in enumerate(dataloader):
            device_batch = self.to_device(batch)
            out = self.model.test_step(device_batch)

            # -- Append every batch
            # TODO: make it general and merrge with collect --> collate
            for key, val in out.items():
                val = val.cpu()
                if to_numpy:
                    val = val.numpy()
                out_dict.setdefault(key, []).append(val)

            pbar.update()
        pbar.clear()
        pbar.close()

        # -- Aggregate on batch dimension
        for key, val in out_dict.items():
            if to_numpy:
                out_dict[key] = np.concatenate(val, axis=0)
            else:
                out_dict[key] = torch.cat(val, dim=0)

        return out_dict
