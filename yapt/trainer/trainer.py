import json
import glob
import re
import os
import torch
import numpy as np

from itertools import cycle, islice

from yapt.utils.trainer_utils import alternate_datasets, to_device
from yapt.utils.utils import stats_to_str, safe_mkdirs
from yapt.utils.utils import is_notebook, is_dict, is_optimizer

# from yapt.trainer.sacred_trainer import SacredTrainer
from yapt.trainer.base import BaseTrainer

if is_notebook():
    from tqdm import tqdm_notebook as tqdm
else:
    from tqdm import tqdm


class Trainer(BaseTrainer):

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

    @property
    def checkpoints_dir(self):
        return os.path.join(self._logdir, 'checkpoints')

    @property
    def checkpoints_format(self):
        return self.args.loggers.checkpoints_format

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
        self.early_stopping = self.get_maybe_missing_args('early_stopping')

    def init_data_loaders(self, data_loaders: dict):

        assert data_loaders.get('train', None) is not None, \
            "train_loader cannot be None"
        self._train_loader = data_loaders['train']
        self._val_loader = data_loaders.get('val', None)
        self._test_loader = data_loaders.get('test', None)
        self.semi_supervised = self.args.semi_supervised

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
            if isinstance(self._train_loader, dict):
                assert 'labelled' in self._train_loader.keys(), \
                    "Train loader dict should contain 'labelled' key"
            else:
                self._train_loader = {'labelled': self._train_loader}

            # -- get train loader len
            self.num_batches_train = self.get_maybe_missing_args('num_batches_train')
            if self.num_batches_train is None:
                self.num_batches_train = len(self._train_loader['labelled'])

        if self._val_loader is not None:
            if not isinstance(self._val_loader, dict):
                self._val_loader = {'validation': self._val_loader}

            # -- get val loaders len
            num_batches_val = self.get_maybe_missing_args('num_batches_val')
            self.num_batches_val = dict()
            for k, v in self._val_loader.items():
                self.num_batches_val[k] = len(v) if num_batches_val is None \
                    else num_batches_val

        if self._test_loader is not None:
            if not isinstance(self._test_loader, dict):
                self._test_loader = {'test': self._test_loader}

            # -- get test loader len
            num_batches_test = self.get_maybe_missing_args('num_batches_test')
            self.num_batches_test = dict()
            for k, v in self._test_loader.items():
                self.num_batches_test[k] = len(v) if num_batches_test is None \
                    else num_batches_test

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
            self.console_log.error(
                "Error occurred while saving results into JSON: %s", e)

    def save_checkpoint(self, path=None, filename=None):
        if filename is None:
            filename = self._epoch

        if isinstance(filename, int):
            filename = self.checkpoints_format.format(filename)

        if path is None:
            path = self.checkpoints_dir

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
            self.console_log.error(
                "Error occurred while saving the checkpoint: %s", e)
        return filename

    def load_checkpoint(self, filename=None):

        path = self.checkpoints_dir
        ckp_format = self.checkpoints_format

        if filename is None:
            filename = os.path.join(path, ckp_format.format(self._epoch))

        elif isinstance(filename, int):
            filename = os.path.join(path, ckp_format.format(filename))

        assert isinstance(filename, str), \
            'filename should be the epoch (int) or the checkpoint path (str)'

        checkpoint = torch.load(filename)
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
        ckp_format = self.checkpoints_format
        if os.path.isfile(self._restart_path):
            # check if restart_path is a specific checkpoint
            self.console_log.info("Reload checkpoint: %s", self._restart_path)
            self.load_checkpoint(self._restart_path)
        else:
            # restore last one
            regex = re.compile(r'.*' + ckp_format.format('(\d+)'))

            _, ext = os.path.splitext(ckp_format)
            checkpoints = glob.glob(os.path.join(
                self._restart_path, "*.{}".format(ext)))

            # Sort checkpoints
            checkpoints = sorted(
                checkpoints, key=lambda f: int(regex.findall(f)[0]))
            last_checkpoint = checkpoints[-1]
            self.console_log.info("Reload checkpoint: %s", last_checkpoint)
            self.load_checkpoint(last_checkpoint)

    def log_each_epoch(self):
        # -- TODO: add anything else to be logged here

        # -- Log learning rates
        optimizers = self._model.optimizer
        if is_optimizer(optimizers):
            current_lr = optimizers.param_groups[0]['lr']
            self._logger.log_metric('optim/lr', current_lr, self._epoch)

        elif is_dict(optimizers):
            for key, opt in optimizers.items():
                current_lr = opt.param_groups[0]['lr']
                self._logger.log_metric(
                    'optim/lr_{}'.format(key), current_lr, self._epoch)
        else:
            raise ValueError(
                "optimizer should be a dict or a torch.optim.Optimizer object")

    def train_epoch(self, dataloader):

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

        pbar_descr_prefix = "ep %d (best: %d beaten: %d) - " % (
            self._epoch, self._best_epoch, self._beaten_epochs)

        self._train_pbar = tqdm(
            dataloader, total=self.num_batches_train,
            desc='', **self.args.loggers.tqdm)

        try:
            # -- Start epoch
            for batch_idx, batch in enumerate(self._train_pbar):
                device_batch = self.to_device(batch)

                # -- Model specific schedulers
                self._model.custom_schedulers()

                # -- Execute a training step
                outputs = self._model.training_step(
                    device_batch, self._epoch)

                # -- Save output for each training step
                # self.collect_outputs(outputs)
                self._global_step = self._model.global_step

                # -- Eventually log statistics
                if self._global_step % self.log_every == 0:
                    self._model.log_train(outputs.get('stats', dict()))
                    self._model.log_grads()

                running_tqdm = outputs.get('running_tqdm', dict())
                # self._train_pbar.set_postfix(ordered_dict=running_tqdm)
                self._train_pbar.set_description("ep %d - %s" % (
                    self.epoch, stats_to_str(running_tqdm)))
                self._train_pbar.update()
                if batch_idx >= self.num_batches_train:
                    break
            self._train_pbar.clear()
            self._train_pbar.close()
            self._epoch += 1

        except KeyboardInterrupt:
            self.console_log.info('Detected KeyboardInterrupt, attempting graceful shutdown...')
            self.shutdown()

        # -- End Epoch
        self._model.reset_train_stats()
        # final_tqdm = self.outputs_train[-1][-1].get('final_tqdm', dict())
        final_tqdm = outputs.get('final_tqdm', dict())
        print(pbar_descr_prefix + stats_to_str(final_tqdm))
        return self.outputs_train[-1]

    def _fit(self):
        """
         A complete training procedure by performing early stopping using the provided validation set
        """

        self.best_epoch_output_train = dict()
        self.best_epoch_output_val = dict()

        if self.early_stopping is not None:
            self.console_log.info(
                "Early stopping: set %s - metric %s - patience %s",
                self.early_stopping.dataset,
                self.early_stopping.metric,
                self.early_stopping.patience)

        # Loads the initial checkpoint if provided
        if self._restart_path is not None:
            self.restart_exp()

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
            # -- Save checkpoint after every epoch
            self.save_checkpoint()
            # -- Validate the network against the validation set
            output_val = dict()
            for kk, vv in self._val_loader.items():
                output_val[kk] = self.validate(
                    vv, self.num_batches_val[kk],
                    set_name=kk, logger=self._logger)
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

        self.console_log.info("Reloading best epoch %d checkpoint", self._best_epoch)
        self.load_checkpoint(self._best_epoch)

        if self._test_loader is not None:
            # -- Test the network
            output_test = dict()
            for kk, vv in self._test_loader.items():
                output_test[kk] = self.validate(
                    vv, self.num_batches_test[kk],
                    set_name=kk, logger=self._logger)
                self.json_results(self._logdir, output_test[kk])
            print("")
        else:
            output_test = output_val[kk]

        return output_test

    def validate(self, dataloader, num_batches, set_name="validation", logger=None):
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

        pbar_descr_prefix = "  %s - " % set_name.title()

        # TODO!!
        # Disable network grad while evaluating the model
        # with no_grad_ifnotscript(self.model):
        if torch.is_grad_enabled():
            self.console_log.warning(
                "You should handle no_grad by yourself for now!")

        # --------------------------------------------------------------------

        self._val_pbar = tqdm(
            dataloader, total=num_batches,
            desc=pbar_descr_prefix, **self.args.loggers.tqdm)

        for batch_idx, batch in enumerate(self._val_pbar):
            device_batch = self.to_device(batch)
            outputs = self._model.validation_step(device_batch, self._epoch)

            running_tqdm = outputs.get('running_tqdm', dict())
            # self._val_pbar.set_postfix(ordered_dict=running_tqdm)
            # self._val_pbar.set_postfix_str(stats_to_str(running_tqdm))
            self._val_pbar.set_description(
                pbar_descr_prefix + stats_to_str(running_tqdm))
            self._val_pbar.update()
            if batch_idx >= num_batches:
                break

        self._val_pbar.clear()
        self._val_pbar.close()

        # --------------------------------------------------------------------

        if logger is not None:
            # NOTE: every key in outputs['stats']
            # is modified with set_name as prefix --> validation/acc
            # this is useful for any logger: tb, neptune, ray
            self._model.log_val(set_name, outputs.get('stats', dict()))

        self._model.reset_val_stats()
        final_tqdm = stats_to_str(outputs.get('final_tqdm', dict()))
        print("  %s - %s" % (set_name.title(), final_tqdm))
        return outputs

    def only_test(self):

        # Reload last or best epoch
        if self._restart_path is not None:
            self.restart_exp()
        else:
            raise ValueError("Give me the folder experiment!")

        self.console_log.info("Reloading best epoch %d checkpoint", self._best_epoch)
        self.load_checkpoint(self._best_epoch)

        if self._test_loader is not None:
            output_test = self.validate(
                self._test_loader, set_name="test", logger=None)
            self.json_results(self._logdir, output_test)
            return output_test

    def test(self, dataloader, to_numpy=True):

        # Load the last / best checkpoint
        if self._restart_path is not None:
            self.restart_exp()
        else:
            raise ValueError("You should specify the experiment folder!")

        print("Reloading best epoch %d checkpoint" % self._best_epoch)
        self.load_checkpoint(self._best_epoch)

        out_dict = {}
        if torch.is_grad_enabled():
            self.console_log.warning("You should handle no_grad by yourself for now!")

        self._test_pbar = tqdm(dataloader, total=self.num_batches_test,
                               desc='test', **self.args.loggers.tqdm)

        for batch_idx, batch in enumerate(self._test_pbar):
            device_batch = self.to_device(batch)
            out = self._model.test_step(device_batch)

            # -- Append every batch
            # TODO: make it general creating a collate function
            for key, val in out.items():
                val = val.cpu()
                if to_numpy:
                    val = val.numpy()
                out_dict.setdefault(key, []).append(val)

            self._test_pbar.update()
            if batch_idx >= self.num_batches_train:
                break

        self._test_pbar.clear()
        self._test_pbar.close()

        # -- Aggregate on batch dimension
        for key, val in out_dict.items():
            if to_numpy:
                out_dict[key] = np.concatenate(val, axis=0)
            else:
                out_dict[key] = torch.cat(val, dim=0)

        return out_dict
