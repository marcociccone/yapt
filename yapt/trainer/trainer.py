import json
import glob
import re
import os
import torch
import numpy as np

from itertools import cycle, islice

from yapt.utils.trainer_utils import alternate_datasets, detach_dict, to_device
from yapt.utils.utils import stats_to_str, safe_mkdirs
from yapt.utils.utils import is_notebook, is_dict

from yapt.trainer.sacred_trainer import SacredTrainer

if is_notebook():
    from tqdm import tqdm_notebook as tqdm
else:
    from tqdm import tqdm


class Trainer(SacredTrainer):

    @property
    def epoch(self):
        return self._epoch

    @property
    def best_epoch(self):
        return self._best_epoch

    @property
    def best_epoch_score(self):
        return self._best_epoch_score

    @property
    def beaten_epochs(self):
        return self._beaten_epochs

    @property
    def train_loader(self):
        return self._train_loader

    @property
    def val_loader(self):
        return self._val_loader

    @property
    def test_loader(self):
        return self._test_loader

    def __init__(self, data_loaders=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        args = self.args

        # -- 1. Load Datasets
        data_loaders = self.set_data_loaders() \
            if data_loaders is None else data_loaders
        self.init_data_loaders(data_loaders)

        self._epoch = 1
        self._best_epoch = 1
        self._best_epoch_score = 0
        self._beaten_epochs = 0
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
        self._train_loader = data_loaders['train']
        self._val_loader = data_loaders.get('val', None)
        self._test_loader = data_loaders.get('test', None)

        self.semi_supervised = self.args.general.semi_supervised

        if self.semi_supervised:
            assert isinstance(self._train_loader, dict), \
                "train_loader should be a dict()"
            assert self._train_loader.get('labelled', None) is not None, \
                "labelled dataloader is None"
            assert self._train_loader.get('unlabelled', None) is not None, \
                "unlabelled dataloader is None"

            self.num_batches_train = max(
                len(self._train_loader['labelled']),
                len(self._train_loader['unlabelled']))
        else:
            self.num_batches_train = len(self._train_loader)
            if isinstance(self._train_loader, dict):
                assert 'labelled' in self._train_loader.keys(), \
                    "Train loader dict should contain 'labelled' key"
            else:
                self._train_loader = {'labelled': self._train_loader}

        if self._val_loader is not None:
            if not isinstance(self._val_loader, dict):
                self._val_loader = {'validation': self._val_loader}

    def json_results(self, savedir, test_score):
        try:
            json_path = os.path.join(savedir, "results.json")
            results = {'global_step': self._global_step,
                       'epoch': self._epoch,
                       'best_epoch': self._best_epoch,
                       'beaten_epochs': self._beaten_epochs,
                       'best_epoch_score': self._best_epoch_score,
                       'test_score': test_score}

            with open(json_path, 'w') as fp:
                json.dump(results, fp)

        except Exception as e:
            print("An error occurred while saving results into JSON:")
            print(e)

    def save_checkpoint(self, path, filename):
        safe_mkdirs(path, exist_ok=True)

        try:
            filename = os.path.join(path, filename)

            current_state_dict = {
                'seen': self._global_step,
                'epoch': self._epoch,
                'best_epoch': self._best_epoch,
                'beaten_epochs': self._beaten_epochs,
                'best_epoch_score': self._best_epoch_score,
                'model_state_dict': self._model.state_dict(),
            }

            # -- there might be more than one optimizer
            if is_dict(self._model.optimizer):
                optimizer_state_dict = {}
                for key, opt in self._model.optimizer.items():
                    optimizer_state_dict.update({key: opt.state_dict()})
            else:
                optimizer_state_dict = self._model.optimizer.state_dict()

            current_state_dict.update(
                {'optimizer_state_dict': optimizer_state_dict})

            torch.save(current_state_dict, filename)

        except Exception as e:
            print("An error occurred while saving the checkpoint:")
            print(e)
        return filename

    def load_checkpoint(self, path, filename):
        ckpt_path = os.path.join(path, filename)

        checkpoint = torch.load(ckpt_path)
        self._global_step = checkpoint.get('global_step', checkpoint.get('seen', 0))
        self._epoch = checkpoint['epoch']
        self._best_epoch = checkpoint['best_epoch']
        self._beaten_epochs = checkpoint['beaten_epochs']
        self._best_epoch_score = checkpoint['best_epoch_score']
        self._model.load_state_dict(checkpoint['model_state_dict'])

        if is_dict(self._model.optimizer):
            for key in self._model.optimizer.keys():
                self._model.optimizer.load_state_dict(
                    checkpoint['optimizer_state_dict'][key])
        else:
            self._model.optimizer.load_state_dict(
                checkpoint['optimizer_state_dict'])

    def restart_exp(self):
        # check if restart_path is a specific checkpoint
        if os.path.isfile(self._restart_path):
            print("Reload checkpoint: %s" % self._restart_path)
            self.load_checkpoint("", self._restart_path)
        # else restore last one
        else:
            regex = re.compile(r'.*epoch(\d+)\.ckpt')
            checkpoints = glob.glob(os.path.join(self._restart_path, "*.ckpt"))
            # Sort checkpoints
            checkpoints = sorted(
                checkpoints, key=lambda f: int(regex.findall(f)[0]))
            last_checkpoint = checkpoints[-1]
            print("Reload checkpoint: %s" % last_checkpoint)
            self.load_checkpoint("", last_checkpoint)

    def log_each_epoch(self):
        # -- TODO: add anything else to be logged here

        # -- Log learning rates
        optimizers = self._model.optimizer
        if isinstance(optimizers, torch.optim.Optimizer):
            current_lr = optimizers.param_groups[0]['lr']
            self._logger.add_scalar('optim/lr', current_lr, self._epoch)

        elif is_dict(optimizers):
            for key, opt in optimizers.items():
                current_lr = opt.param_groups[0]['lr']
                self._logger.add_scalar(
                    'optim/lr_{}'.format(key), current_lr, self._epoch)
        else:
            raise ValueError(
                "optimizer should be a dict or a \
                torch.optim.Optimizer object")

    def train_epoch(self, dataloader):
        """
        Performs one entire epoch of training
        :param dataloader: A DataLoader object producing training samples
        :return: a tuple (epoch_loss, epoch_accuracy)
        """

        # -- Call Optimizer schedulers and track lr
        if self._epoch > 1:
            # -- Pytorch 1.1.0 requires to call first optimizer.step()
            self.call_schedulers_optimizers()
        self.log_each_epoch()

        # -- Initialization and training mode
        # Enters train mode
        self._model.train()
        # Zero the parameter gradients
        self._model.zero_grad()
        # Track training statistist
        self._model.init_train_stats()
        self.outputs_train.append([])

        pbar_descr_prefix = "Epoch %d (best: %d, beaten: %d) - " % (
            self._epoch, self._best_epoch, self._beaten_epochs)

        pbar = tqdm(total=self.num_batches_train,
                    desc=pbar_descr_prefix,
                    **self.args.tqdm)

        # -- Start epoch
        for batch_idx, batch in enumerate(dataloader):
            device_batch = self.to_device(batch)

            # -- Model specific schedulers
            self._model.custom_schedulers(
                self._epoch, logger=self._logger)

            # -- Execute a training step
            outputs = self._model.training_step(
                device_batch, self._epoch)

            # -- Save output for each training step
            self.collect_outputs(outputs)
            self._global_step = self._model.global_step

            # -- Eventually log statistics on tensorboard
            if self._global_step % self.log_every == 0:
                self._model.log_train(
                    outputs.get('stats', dict()), self._logger)

            running_tqdm = stats_to_str(outputs.get('running_tqdm', dict()))
            pbar.set_description(pbar_descr_prefix + running_tqdm)
            pbar.update()

        pbar.clear()
        pbar.close()
        self._epoch += 1

        # -- End Epoch
        self._model.reset_train_stats()
        final_tqdm = self.outputs_train[-1][-1].get('final_tqdm', dict())
        print(pbar_descr_prefix + stats_to_str(final_tqdm))
        return self.outputs_train[-1]

    def _fit(self):
        """
         A complete training procedure by performing early stopping using the provided validation set
        """

        self.best_epoch_output_train = dict()
        self.best_epoch_output_val = dict()

        if self.early_stopping is not None:
            self.print_verbose(
                "Early stopping: set {} - metric {} - patience {}".format(
                    self.early_stopping.dataset,
                    self.early_stopping.metric,
                    self.early_stopping.patience))

        # Loads the initial checkpoint if provided
        if self._restart_path is not None:
            self.restart_exp()

        self.log_args()

        while self._epoch < self.max_epochs:

            if (self.early_stopping is not None and
                self._beaten_epochs > self.early_stopping.patience):
                break

            # -- TODO: move out
            if self.semi_supervised:
                if self.args.alternated_update:
                    train_loader = alternate_datasets(
                        iter(self._train_loader['labelled']),
                        iter(self._train_loader['unlabelled']))
                else:
                    train_loader = zip(
                        islice(cycle(self._train_loader['labelled']),
                               self.num_batches_train),
                        self._train_loader['unlabelled'])
            else:
                train_loader = self._train_loader['labelled']

            # -- Performs one epoch of training
            output_train = self.train_epoch(train_loader)
            self.save_checkpoint(self._logdir, "epoch%d.ckpt" % self._epoch)

            # -- Validate the network against the validation set
            output_val = dict()
            for kk, vv in self._val_loader.items():
                output_val[kk] = self.validate(
                    vv, log_descr=kk, logger=self._logger)
            print("")

            is_best_epoch, best_score = self._model.early_stopping(
                output_val, self.best_epoch_output_val)

            if is_best_epoch or self._epoch == 1:
                self._beaten_epochs = 0
                self._best_epoch = self._epoch
                self._best_epoch_score = best_score
                self.best_epoch_output_train = output_train
                self.best_epoch_output_val = output_val
            else:
                self._beaten_epochs += 1

        print("Reloading best epoch %d checkpoint" % self._best_epoch)
        self.load_checkpoint(self._logdir, "epoch%d.ckpt" % self._best_epoch)

        if self._test_loader is not None:
            # TODO !!!
            output_test = self.validate(
                self._test_loader, log_descr="test", logger=None)
            self.json_results(self._logdir, output_test)
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
        self._model.eval()
        self._model.init_val_stats()

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
            outputs = self._model.validation_step(device_batch, self._epoch)

            running_tqdm = stats_to_str(outputs.get('running_tqdm', dict()))
            pbar.set_description(pbar_descr_prefix + running_tqdm)
            pbar.update()

        pbar.clear()
        pbar.close()

        # --------------------------------------------------------------------

        if logger is not None:
            self._model.log_val(
                self._epoch, log_descr,
                outputs.get('stats', dict()), logger)

        self._model.reset_val_stats()
        final_tqdm = stats_to_str(outputs.get('final_tqdm', dict()))
        print("\t%s - %s" % (log_descr.title(), final_tqdm))
        return outputs

    def only_test(self):

        # Reload last or best epoch
        if self._restart_path is not None:
            self.restart_exp()
        else:
            raise ValueError("Give me the folder experiment!")

        print("Reloading best epoch %d checkpoint" % self._best_epoch)
        self.load_checkpoint(self._logdir, "epoch%d.ckpt" % self._best_epoch)

        if self._test_loader is not None:
            output_test = self.validate(
                self._test_loader, log_descr="test", logger=None)
            self.json_results(self._logdir, output_test)
            return output_test

    def test(self, dataloader, to_numpy=True):

        # Load the last / best checkpoint
        if self._restart_path is not None:
            self.restart_exp()
        else:
            raise ValueError("Give me the folder experiment!")

        print("Reloading best epoch %d checkpoint" % self._best_epoch)
        self.load_checkpoint(self._logdir, "epoch%d.ckpt" % self._best_epoch)

        out_dict = {}
        if torch.is_grad_enabled():
            print("WARNING: You should handle no_grad by yourself for now!")

        pbar = tqdm(total=len(dataloader), desc='test',
                    **self.args.tqdm)

        for batch_idx, batch in enumerate(dataloader):
            device_batch = self.to_device(batch)
            out = self._model.test_step(device_batch)

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
