import glob
import re
import os
import torch
import numpy as np

from itertools import cycle, islice
from yapt.utils.utils import is_notebook, is_dict, is_optimizer, stats_to_str
from yapt.utils.storage import safe_mkdirs
from yapt.utils.debugging import timing
from yapt.utils.torch_helpers import alternate_datasets
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
        return self._model._best_epoch

    @property
    def best_epoch_score(self):
        return self._model._best_epoch_score

    @property
    def best_stats(self):
        return self._model._best_stats

    @property
    def beaten_epochs(self):
        return self._model._beaten_epochs

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
    def debug_dir(self):
        _debug_dir = os.path.join(self._logdir, 'debug')
        safe_mkdirs(_debug_dir, True)
        return _debug_dir

    @property
    def inputs_dir(self):
        _inputs_dir = os.path.join(self.debug_dir, 'inputs')
        safe_mkdirs(_inputs_dir, True)
        return _inputs_dir

    @property
    def checkpoints_dir(self):
        _checkpoints_dir = os.path.join(self._logdir, 'checkpoints')
        safe_mkdirs(_checkpoints_dir, True)
        return _checkpoints_dir

    @property
    def best_checkpoints_dir(self):
        _checkpoints_dir = os.path.join(self.checkpoints_dir, 'best')
        safe_mkdirs(_checkpoints_dir, True)
        return _checkpoints_dir

    @property
    def results_dir(self):
        _results_dir = os.path.join(self._logdir, 'results')
        safe_mkdirs(_results_dir, True)
        return _results_dir

    @property
    def images_dir(self):
        _results_dir = os.path.join(self._logdir, 'images')
        safe_mkdirs(_results_dir, True)
        return _results_dir

    @property
    def videos_dir(self):
        _results_dir = os.path.join(self._logdir, 'videos')
        safe_mkdirs(_results_dir, True)
        return _results_dir

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

        self._epoch = 0
        # -- Early Stopping
        self.save_every = args.loggers.save_every
        self.validate_every = args.loggers.validate_every
        self.keep_only_last_checkpoint = args.loggers.keep_only_last_checkpoint
        self.last_checkpoint = None
        self.max_epochs = args.max_epochs
        self.early_stopping = self.get_maybe_missing_args('early_stopping')

    def init_data_loaders(self, data_loaders: dict):

        assert data_loaders.get('train', None) is not None, \
            "train_loader cannot be None"
        self._train_loader = data_loaders['train']
        self._val_loader = data_loaders.get('val', None)
        self._test_loader = data_loaders.get('test', None)
        self.semi_supervised = self.args.data.semi_supervised
        self.alternated_update = self.args.data.alternated_update

        if self.semi_supervised:
            assert isinstance(self._train_loader, dict), \
                "train_loader should be a dict()"
            assert self._train_loader.get('labelled', None) is not None, \
                "labelled dataloader is None"
            assert self._train_loader.get('unlabelled', None) is not None, \
                "unlabelled dataloader is None"

            if self.alternated_update:
                self.num_batches_train = (
                    len(self._train_loader['labelled']) +
                    len(self._train_loader['unlabelled']))
            else:
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
        else:
            self._val_loader = dict()

        if self._test_loader is not None:
            if not isinstance(self._test_loader, dict):
                self._test_loader = {'test': self._test_loader}

            # -- get test loader len
            num_batches_test = self.get_maybe_missing_args('num_batches_test')
            self.num_batches_test = dict()
            for k, v in self._test_loader.items():
                self.num_batches_test[k] = len(v) if num_batches_test is None \
                    else num_batches_test
        else:
            self._test_loader = dict()

    def get_train_loader(self):
        if self.semi_supervised:
            if self.alternated_update:
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

        return train_loader

    @timing
    def save_results(self, outputs, name, idx=''):
        try:
            path = os.path.join(self.results_dir, name)
            filename = os.path.join(path, "%s%s.pt" % (name, idx))
            safe_mkdirs(path)

            results = {
                'global_step': self.global_step,
                'epoch': self.epoch,
                'best_epoch': self.best_epoch,
                'beaten_epochs': self.beaten_epochs,
                'best_epoch_score': self.best_epoch_score,
                'best_stats': self.best_stats,
                'outputs': outputs
            }
            torch.save(results, filename)

        except Exception as e:
            self.console_log.error(
                "Error occurred while saving results into : %s", e)

    @timing
    def save_checkpoint(self, path=None, filename=None, is_best=False):
        if filename is None:
            filename = self._epoch

        if isinstance(filename, int):
            filename = self.checkpoints_format.format(filename)

        if path is None:
            path = self.checkpoints_dir

        if is_best:
            path = self.best_checkpoints_dir

        safe_mkdirs(path, exist_ok=True)

        try:
            filename = os.path.join(path, filename)

            current_state_dict = {
                'global_step': self._global_step,
                'epoch': self._epoch,
                'best_epoch': self.best_epoch,
                'beaten_epochs': self.beaten_epochs,
                'best_epoch_score': self.best_epoch_score,
                'best_stats': self.best_stats,
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

            # -- track filename to delete after,
            # if keep_only_last_checkpoint is set true
            if not is_best and 'init' not in filename:
                self.last_checkpoint = filename

        except Exception as e:
            self.console_log.error(
                "Error occurred while saving the checkpoint: %s", e)
        return filename

    def load_checkpoint(self, filename=None, is_best=False):
        """
            This function actually restores a checkpoint with:

            - model state_dict
            - optimizers state_dict
            - global_step
            - epoch
            - best_epoch
            - beaten_epochs
            - best_epoch_score
            - best_stats
        """

        path = self.checkpoints_dir
        ckp_format = self.checkpoints_format
        if is_best:
            path = self.best_checkpoints_dir
        if filename is None:
            filename = os.path.join(path, ckp_format.format(self._epoch))

        elif isinstance(filename, int):
            filename = os.path.join(path, ckp_format.format(filename))

        assert isinstance(filename, str), \
            'filename should be the epoch (int) or the checkpoint path (str)'

        checkpoint = torch.load(filename)
        self._global_step = checkpoint.get('global_step', checkpoint.get('seen', 0))
        self._epoch = checkpoint['epoch']

        self._model.best_epoch = checkpoint.get('best_epoch', -1)
        self._model.beaten_epochs = checkpoint.get('beaten_epochs', 0)
        self._model.best_epoch_score = checkpoint.get('best_epoch_score', 0)
        self._model.best_stats = checkpoint.get('best_stats', [])

        self._model.load_state_dict(checkpoint['model_state_dict'])

        if is_dict(self._model.optimizer):
            for key in self._model.optimizer.keys():
                self._model.optimizer.load_state_dict(
                    checkpoint['optimizer_state_dict'][key])
        else:
            self._model.optimizer.load_state_dict(
                checkpoint['optimizer_state_dict'])

    def load_checkpoint_for_test(self):
        """
            This function can be used at the end of a training,
            because it needs best_epoch and (current) epoch to be up to date.
        """

        if self.best_epoch > 0:
            # -- if taking track of best model
            self.console_log.info("Reloading best epoch %d ", self.best_epoch)
            self.load_checkpoint(self.best_epoch, is_best=True)
        else:
            self.console_log.info("Reloading last epoch %d ", self.epoch)
            self.load_checkpoint(self.epoch, is_best=False)

    def restore_exp(self):
        """
            If a restore_path is specified it restore a specific checkpoint.
            Otherwise, it finds the last checkpoint in checkpoints_dir and
            reload it.
        """

        ckp_format = self.checkpoints_format
        if os.path.isfile(self._restore_path):
            # check if restart_path is a specific checkpoint
            self.console_log.info("Reload checkpoint: %s", self._restore_path)
            self.load_checkpoint(self._restore_path)
        else:
            # restore last one
            regex = re.compile(r'.*' + ckp_format.format('(\d+)'))

            _, ext = os.path.splitext(ckp_format)
            checkpoints = glob.glob(os.path.join(
                self._restore_path, 'checkpoints', "*{}".format(ext)))
            checkpoints = list(filter(lambda x: 'init' not in x, checkpoints))

            # Sort checkpoints
            checkpoints = sorted(
                checkpoints, key=lambda f: int(regex.findall(f)[0]))
            last_checkpoint = checkpoints[-1]
            self.console_log.info("Reload checkpoint: %s", last_checkpoint)
            self.load_checkpoint(last_checkpoint)

    def save_last_checkpoint(self):
        last_checkpoint = self.last_checkpoint
        self.save_checkpoint()
        if self.keep_only_last_checkpoint and last_checkpoint is not None:
            try:
                os.remove(last_checkpoint)
            except OSError:
                pass

    def clean_best_checkpoints(self):
        """
        This function keeps only topk best performing checkpoints.
        Delete all the rest from file system.
        Use `args.loggers.keep_topk_checkpoints` to control topk.
        """

        topk = self.args.loggers.keep_topk_checkpoints
        for el in self.best_stats[:-topk]:
            filename = os.path.join(
                self.best_checkpoints_dir,
                self.checkpoints_format.format(el[0]))
            try:
                os.remove(filename)
            except OSError:
                pass

    def save_best_checkpoint(self, is_best):
        if is_best:
            self.save_checkpoint(is_best=True)
            self.clean_best_checkpoints()

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

    @timing
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
        self._model.zero_grad.calls -= 1
        # Track training statistist
        self._model.init_train_stats()

        pbar_descr_prefix = "ep %d (best: %d beaten: %d) - " % (
            self._epoch, self.best_epoch, self.beaten_epochs)

        self._train_pbar = tqdm(
            dataloader, total=self.num_batches_train,
            desc='', **self.args.loggers.tqdm)
        try:
            # -- Start epoch
            outputs = None
            accum_stats = []
            self._model.on_epoch_start()
            for batch_idx, batch in enumerate(self._train_pbar):
                if batch_idx >= self.num_batches_train:
                    break

                if self.args.debug.save_inputs:
                    # epoch
                    path = os.path.join(self.inputs_dir, 'epoch_{}'.format(self.epoch))
                    safe_mkdirs(path, exist_ok=True)
                    filename = os.path.join(path, 'batch_{}.pt'.format(batch_idx))
                    torch.save(batch, filename)

                device_batch = self.to_device(batch)

                # -- Model specific schedulers
                self._model.custom_schedulers()

                # -- Execute a training step
                outputs = self._model.training_step(
                    device_batch)

                # -- Accumulate stats from grads accum steps
                accum_stats.append(outputs.get('stats', dict()))

                # -- Save output for each training step
                # self.collect_outputs(outputs)

                if self._model._train_step % self.args.accum_batches == 0:
                    # -- Increment the global step only
                    # every accum_batches batches
                    self._global_step += 1

                    # -- Aggregate stats from grads accum steps
                    stats = self._model.aggregate_accum_stats(accum_stats)
                    accum_stats = []

                    # -- Eventually log statistics
                    if self._global_step % self.log_every == 0:
                        self._model.log_train(stats)
                        self._model.log_grads()

                running_tqdm = outputs.get('running_tqdm', dict())
                # self._train_pbar.set_postfix(ordered_dict=running_tqdm)
                self._train_pbar.set_description("ep %d - %s" % (
                    self.epoch, stats_to_str(running_tqdm)))
                self._train_pbar.update()

            self.console_log.info("Processed {} batches.".format(batch_idx))
            self._train_pbar.clear()
            self._train_pbar.close()

        except KeyboardInterrupt:
            self.console_log.info('Detected KeyboardInterrupt, attempting graceful shutdown...')
            self.shutdown()

        # -- End Epoch
        self._model.on_epoch_end()
        self._model.reset_train_stats()

        if outputs is not None:
            final_tqdm = outputs.get('final_tqdm', dict())
            print(pbar_descr_prefix + stats_to_str(final_tqdm))
        return outputs

    def _fit(self):
        """
         A complete training procedure by performing early stopping using the provided validation set
        """

        if self.early_stopping is not None:
            self.console_log.info(
                "Early stopping: set %s - metric %s - patience %s - mode %s",
                self.early_stopping.dataset,
                self.early_stopping.metric,
                self.early_stopping.patience,
                self.early_stopping.mode
            )

        # Reload last epoch
        if self._restore_path is not None:
            self.restore_exp()
        else:
            # -- Save initialized weights: it could be useful for debugging
            self.save_checkpoint(filename=self.checkpoints_format.format('init'))

        self._model.on_train_start()
        while self._epoch < self.max_epochs:

            self._epoch += 1
            # -- If patience has been reached
            if self._model.early_stop:
                break

            # -- Get loader for supervised of semi-supervised
            train_loader = self.get_train_loader()

            # -- Performs one epoch of training
            output_train = self.train_epoch(train_loader)

            # -- Save checkpoint and results after every epoch
            if self._epoch % self.save_every == 0:
                self.save_results(output_train, 'train', self._epoch)
                self.save_last_checkpoint()

            # -- Validate the model against the validation set
            if self._epoch % self.validate_every == 0:
                output_val = dict()
                for kk, vv in self._val_loader.items():
                    output_val[kk] = self.validate(
                        vv, self.num_batches_val[kk],
                        set_name=kk, logger=self._logger)
                    self.save_results(output_val[kk], kk, self.epoch)
                print("")

                is_best, best_score = self._model.early_stopping(
                    output_val)
                self.save_best_checkpoint(is_best)

        self._model.on_train_end()
        # -- reload last or best checkpoint
        self.load_checkpoint_for_test()
        if self._test_loader is not None:
            # -- Test the network
            output_test = dict()
            for kk, vv in self._test_loader.items():
                output_test[kk] = self.validate(
                    vv, self.num_batches_test[kk],
                    set_name=kk, logger=self._logger)
                self.save_results(output_test[kk], kk, self.epoch)
            print("")
        else:
            output_test = output_val[kk]

        return output_test

    @timing
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
        self._model.on_validation_start(set_name)

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

        outputs = None
        outputs_list = []
        for batch_idx, batch in enumerate(self._val_pbar):
            if batch_idx >= num_batches:
                break
            device_batch = self.to_device(batch)
            outputs = self._model.validation_step(device_batch)

            # -- accumulate outputs from validation steps to be used later
            outputs_list.append(outputs)

            running_tqdm = outputs.get('running_tqdm', dict())
            # self._val_pbar.set_postfix(ordered_dict=running_tqdm)
            # self._val_pbar.set_postfix_str(stats_to_str(running_tqdm))
            self._val_pbar.set_description(
                pbar_descr_prefix + stats_to_str(running_tqdm))
            self._val_pbar.update()

        self._val_pbar.clear()
        self._val_pbar.close()

        # --------------------------------------------------------------------

        self._model.on_validation_end(set_name, outputs_list)
        self._model.reset_val_stats()
        if outputs is not None:
            if logger is not None:
                # NOTE: every key in outputs['stats']
                # is modified with set_name as prefix --> validation/acc
                # this is useful for any logger: tb, neptune, ray
                self._model.log_val(set_name, outputs.get('stats', dict()))

            final_tqdm = stats_to_str(outputs.get('final_tqdm', dict()))
            print("  %s - %s" % (set_name.title(), final_tqdm))

        return outputs

    def only_test(self):

        # Reload last or best epoch
        if self._restore_path is not None:
            self.restore_exp()
        else:
            raise ValueError("Experiment folder is missing!")

        self.console_log.info("Reloading best epoch %d checkpoint", self.best_epoch)
        self.load_checkpoint(self.best_epoch, is_best=True)

        if self._test_loader is not None:
            # -- Test the network
            output_test = dict()
            for kk, vv in self._test_loader.items():
                output_test[kk] = self.validate(
                    vv, self.num_batches_test[kk],
                    set_name=kk, logger=self._logger)
                self.save_results(output_test[kk], kk)
            return output_test

    def test(self, dataloader, num_batches, set_name="test", to_numpy=True):

        # Load the last / best checkpoint
        if self._restore_path is not None:
            self.restore_exp()
        else:
            raise ValueError("You should specify the experiment folder!")

        print("Reloading best epoch %d checkpoint" % self._best_epoch)
        self.load_checkpoint(self.best_epoch, is_best=True)

        out_dict = {}
        if torch.is_grad_enabled():
            self.console_log.warning("You should handle no_grad by yourself for now!")

        self._test_pbar = tqdm(dataloader, total=num_batches,
                               desc=set_name, **self.args.loggers.tqdm)

        for batch_idx, batch in enumerate(self._test_pbar):
            if batch_idx >= num_batches:
                break
            device_batch = self.to_device(batch)
            out = self._model.test_step(device_batch)

            # -- Append every batch
            # TODO: make it general creating a collate function
            for key, val in out.items():
                if torch.is_tensor(val):
                    val = val.cpu()
                    if to_numpy:
                        val = val.numpy()
                out_dict.setdefault(key, []).append(val)

            self._test_pbar.update()

        self._test_pbar.clear()
        self._test_pbar.close()

        # -- Aggregate on batch dimension
        for key, val in out_dict.items():
            if to_numpy:
                out_dict[key] = np.concatenate(val, axis=0)
            else:
                out_dict[key] = torch.cat(val, dim=0)

        return out_dict
