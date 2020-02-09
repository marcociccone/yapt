import os
import re
import sys
import glob
import json
import torch
import inspect
import functools
import numpy as np

from itertools import cycle, islice
from tqdm import tqdm
from time import gmtime, strftime

from sacred import Experiment
from sacred.observers import MongoObserver
from sacred.utils import apply_backspaces_and_linefeeds

from yapt.utils.trainer_utils import alternate_datasets, detach_dict, to_device
from yapt.utils.utils import safe_mkdirs
from yapt.core.logger.tensorboardXsafe import SummaryWriter

class DisableGradNotScriptContext:
    def __init__(self, model):
        self.model = model
        self.script = self.has_scriptmodule(self.all_submodules(model))
        self.context = None

    @staticmethod
    def all_submodules(module):
        submod = list(module.modules())
        for m in module.modules():
            if m != module:
                submod += DisableGradNotScriptContext.all_submodules(m)
        return submod

    @staticmethod
    def has_scriptmodule(module_list):
        for mod in module_list:
            if isinstance(mod, torch.jit.ScriptModule):
                return True
        return False

    def __enter__(self):
        if not self.script:
            self.context = torch.no_grad()
            self.context.__enter__()

    def __exit__(self, *args):
        if not self.script:
            self.context.__exit__(*args)

# Rename class for convenience
no_grad_ifnotscript = DisableGradNotScriptContext


def main_ifsacred(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        self = args[0]
        # Check at run-time if sacred has to be used
        if self.use_sacred:
            # If this is the case, wrap the function
            @self.sacred_exp.main
            def decor_func():
                return func(*args, **kwargs)

            # and run it through sacred run()
            self.sacred_exp.run()
        else:
            # Otherwise just run the function
            return func(*args, **kwargs)

    return wrapper


def _is_method(obj, name):
    return hasattr(obj, name) and inspect.ismethod(getattr(obj, name))


class Trainer:
    def __init__(self,
                 args,
                 model_class:None,
                 data_loaders=None,
                 lr_scheduler=None,
                 params_scheduler=None,
                 init_seeds=True,
                 ):

        super().__init__()
        self.verbose = True
        self.init_data = False
        self.args = args
        self.use_cuda = args.cuda and torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")

        # -- 0. Init random seed
        self.init_seeds(init_seeds)

        # -- 1. Load Datasets
        data_loaders = self.set_data_loaders() \
            if data_loaders is None else data_loaders
        self.init_data_loaders(data_loaders)

        # -- 2. Load Model
        model = self.set_model() if model_class is None else model_class(args)
        self.model = model.to(self.device)

        # -- Optimization # TODO: Move inside model
        self.lr_scheduler = lr_scheduler
        self.params_scheduler = params_scheduler

        # -- Early Stopping
        self.max_epochs = args.max_epochs
        self.early_stopping = args.early_stopping
        self.patience = args.early_stopping.patience
        self.restart_path = args.restart_path

        # -- Logging
        self.log_every = args.logging.log_every
        self.use_sacred = args.logging.sacred
        if self.use_sacred:
            self.sacred_exp = Experiment(args.exp_name)
            self.sacred_exp.captured_out_filter = apply_backspaces_and_linefeeds
            self.sacred_exp.add_config(vars(args))
            for source in self.get_sources():
                self.sacred_exp.add_source_file(source)

            if not args.logging.mongodb_disable:
                url = "{0.mongodb_url}:{0.mongodb_port}".format(args)
                if (args.logging.mongodb_name is not None and
                        args.logging.mongodb_name != ''):
                    db_name = args.logging.mongodb_name
                else:
                    db_name = 'ciccone_' + ''.join(filter(
                        str.isalnum, args.dataset_name.lower()))

                print('Connect to MongoDB@{}:{}'.format(url, db_name))
                self.sacred_exp.observers.append(MongoObserver.create(url=url, db_name=db_name))

        self.seen = 0
        self.epoch = 0
        self.best_epoch = 0
        self.best_epoch_score = 0
        self.beaten_epochs = 0
        self.best_epoch_output_train = dict()
        self.best_epoch_output_val = dict()

        self.outputs_train = list()
        self.outputs_val = list()

        timestring = strftime("%Y-%m-%d_%H-%M-%S", gmtime()) + "_%s" % args.exp_name
        if self.restart_path != '' and not self.args.use_new_dir:
            if os.path.isfile(self.restart_path):
                self.logdir = os.path.dirname(self.restart_path)
            elif os.path.isdir(self.restart_path):
                self.logdir = self.restart_path
            else:
                raise NotImplementedError(
                    "Restart Path %s is not a file nor a dir" %
                    self.restart_path)
        else:
            self.logdir = os.path.join(
                args.logging.logdir, args.dataset_name.lower(), timestring)

    def print_verbose(self, message):
        if self.verbose:
            print(message)

    def init_seeds(self, init_seeds):
        # -- This might be the case the user wants to init the seeds by himself
        # -- For instance, he creates the datasets outside the constructor
        if not init_seeds:
            return

        args = self.args
        if args.seed != -1:
            torch.manual_seed(args.seed)
            torch.cuda.manual_seed(args.seed)
            np.random.seed(args.seed)  # Numpy module.
            # random.seed(args.seed)  # Python random module.
            self.print_verbose("Random seed: {}".format(args.seed))

        torch.backends.cudnn.benchmark = args.cudnn.benchmark
        torch.backends.cudnn.deterministic = args.cudnn.deterministic
        self.print_verbose("cudnn.benchmark: {}".format(args.cudnn.benchmark))
        self.print_verbose("cudnn.deterministic: {}".format(args.cudnn.deterministic))

    def set_data_loaders(self):
        raise NotImplementedError("Implement this method to return a dict \
                                   of dataloaders or pass it to the constructor")

    def set_model(self):
        raise NotImplementedError("Implement this method to return your model \
                                   or pass it to the constructor")

    def init_data_loaders(self, data_loaders: dict):
        if self.init_data:
            return

        assert data_loaders.get('train', None) is not None, \
            "train_loader cannot be None"
        self.train_loader = data_loaders['train']
        self.val_loader = data_loaders.get('val', None)
        self.test_loader = data_loaders.get('test', None)
        self.semi_supervised = self.args.semi_supervised

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
            # else:
            #     assert 'validation' in self.val_loader.keys(), \
            #         "`validation` key dataset not found"

        self.init_data = True

    def get_sources(self):
        sources = []
        # The network file
        sources.append(inspect.getfile(self.model.__class__))
        # the main script
        sources.append(sys.argv[0])
        # and any user custom submodule
        for module in self.model.children():
            module_path = inspect.getfile(module.__class__)
            if 'site-packages' not in module_path:
                sources.append(module_path)
        return sources

    def log_sacred_scalar(self, name, val, step):
        if self.use_sacred and self.sacred_exp.current_run:
            self.sacred_exp.current_run.log_scalar(name, val, step)

    def log_params(self, logger):
        name_str = os.path.basename(sys.argv[0])
        args_str = "".join([("%s: %s, " % (arg, val)) for arg, val in sorted(vars(self.args).items())])[:-2]
        logger.add_text("Script arguments", name_str + " -> " + args_str)

    def json_params(self, savedir):
        try:
            dict_params = vars(self.args)
            json_path = os.path.join(savedir, "args.json")

            with open(json_path, 'w') as fp:
                json.dump(dict_params, fp)
        except Exception as e:
            print("An error occurred while saving parameters into JSON:")
            print(e)

    def json_results(self, savedir, test_score):
        try:
            json_path = os.path.join(savedir, "results.json")
            results = {'seen': self.seen,
                       'epoch': self.epoch,
                       'best_epoch': self.best_epoch,
                       'beaten_epochs': self.beaten_epochs,
                       'best_epoch_score': self.best_epoch_score,
                       'test_score': test_score}

            with open(json_path, 'w') as fp:
                json.dump(results, fp)

            if self.use_sacred and self.sacred_exp.current_run:
                self.sacred_exp.current_run.current_run.add_artifact(json_path)

        except Exception as e:
            print("An error occurred while saving results into JSON:")
            print(e)

    def log_gradients(self, logger, global_step):

        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                logger.add_scalar("gradients/"+name, param.grad.norm(2).item(),
                                  global_step=global_step)

    def restart_exp(self):
        # check if restart_path is a specific checkpoint
        if os.path.isfile(self.restart_path):
            print("Reload checkpoint: %s" % self.restart_path)
            self.load_checkpoint("", self.restart_path)
        # else restore last one
        else:
            regex = re.compile(r'.*epoch(\d+)\.ckpt')
            checkpoints = glob.glob(os.path.join(self.restart_path, "*.ckpt"))
            # Sort checkpoints
            checkpoints = sorted(
                checkpoints, key=lambda f: int(regex.findall(f)[0]))
            last_checkpoint = checkpoints[-1]
            print("Reload checkpoint: %s" % last_checkpoint)
            self.load_checkpoint("", last_checkpoint)

    def save_checkpoint(self, path, filename):
        safe_mkdirs(path, exist_ok=True)

        try:
            torch.save({'seen': self.seen,
                        'epoch': self.epoch,
                        'best_epoch': self.best_epoch,
                        'beaten_epochs': self.beaten_epochs,
                        'best_epoch_score': self.best_epoch_score,
                        'model_state_dict': self.model.state_dict(),
                        }, os.path.join(path, filename))

        except Exception as e:
            print("An error occurred while saving the checkpoint:")
            print(e)

    def load_checkpoint(self, path, filename):
        ckpt_path = os.path.join(path, filename)

        checkpoint = torch.load(ckpt_path)
        self.seen = checkpoint['seen']
        self.epoch = checkpoint['epoch']
        self.best_epoch = checkpoint['best_epoch']
        self.beaten_epochs = checkpoint['beaten_epochs']
        self.best_epoch_score = checkpoint['best_epoch_score']
        self.model.load_state_dict(checkpoint['model_state_dict'])

    def only_test(self):
        # Loads the last checkpoint
        if self.restart_path != '':
            self.restart_exp()
        else:
            raise ValueError("Give me the folder experiment!")

        print("Reloading best epoch %d checkpoint" % self.best_epoch)
        self.load_checkpoint(self.logdir, "epoch%d.ckpt" % self.best_epoch)

        if self.test_loader is not None:
            output_test = self.validate(
                self.test_loader, log_descr="test", logger=None)
            self.json_results(self.logdir, output_test)
            # self.log_sacred_scalar("test/accuracy", , self.epoch)
            # (test_acc, test_precision,
            #  test_recall, test_f1,
            #  test_auc) = self.model.final_test(self.test_loader)
            # print('test acc {:.2f}, prec {:.2f}, recall {:.2f}, f1 {:.2f}, roc_auc {:.2f}\n'.format(
            #     test_acc, test_precision, test_recall, test_f1, test_auc))
            return output_test

    def test(self, dataloader, to_numpy=True):

        # Loads the last checkpoint
        if self.restart_path != '':
            self.restart_exp()
        else:
            raise ValueError("Give me the folder experiment!")

        print("Reloading best epoch %d checkpoint" % self.best_epoch)
        self.load_checkpoint(self.logdir, "epoch%d.ckpt" % self.best_epoch)

        out_dict = {}
        with torch.no_grad():
            with tqdm(total=len(dataloader), ncols=80, dynamic_ncols=False,
                      desc='test', leave=False) as pbar:

                for batch_idx, batch in enumerate(dataloader):
                    device_batch = self.to_device(batch)
                    out = self.model.test_step(
                        device_batch)

                    # -- Append every batch
                    for key, val in out.items():
                        val = val.cpu()
                        if to_numpy:
                            val = val.numpy()
                        out_dict.setdefault(key, []).append(val)

                    pbar.update()

        # -- Aggregate on batch dimension
        for key, val in out_dict.items():
            if to_numpy:
                out_dict[key] = np.concatenate(val, axis=0)
            else:
                out_dict[key] = torch.cat(val, dim=0)

        return out_dict

    @main_ifsacred
    def fit(self):
        """
         A complete training procedure by performing early stopping using the provided validation set
        """

        self.seen = 0
        self.epoch = 1
        self.best_epoch = 0
        self.beaten_epochs = 0
        self.best_epoch_score = 0
        self.best_epoch_output_train = dict()
        self.best_epoch_output_val = dict()

        # Loads the initial checkpoint if provided
        if self.restart_path != '':
            self.restart_exp()

        safe_mkdirs(self.logdir, exist_ok=True)
        logger = SummaryWriter(log_dir=self.logdir)
        self.log_params(logger)
        self.json_params(self.logdir)

        print("Early stopping: set {} - metric {} - patience {}".format(
            self.early_stopping.dataset,
            self.early_stopping.metric,
            self.early_stopping.patience
        ))

        while (self.beaten_epochs < self.patience and
               self.epoch < self.max_epochs):

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            if self.params_scheduler is not None:
                self.params_scheduler.step()

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

            # -- Performs one epoch of training
            output_train = self.train_epoch(train_loader, logger=logger)
            self.save_checkpoint(self.logdir, "epoch%d.ckpt" % self.epoch)

            # -- Validate the network against the validation set

            output_val = dict()
            for kk, vv in self.val_loader.items():
                output_val[kk] = self.validate(
                    vv, log_descr=kk, logger=logger)
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
                self.test_loader, log_descr="test", logger=logger)
            self.json_results(self.logdir, output_test)
            # self.log_sacred_scalar("test/accuracy", , self.epoch)
        else:
            output_test = output_val[kk]

        return output_test

    # def log_each_epoch(self):
    #     current_lr = self.optimizer.param_groups[0]['lr']
    #     print('current lr {:.5e}'.format(current_lr))
    #     self.logger.add_scalar('train/lr', current_lr, self.epoch)

    def train_epoch(self, dataloader, logger=None):
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

        with tqdm(total=self.num_batches_train,
                  ncols=80, dynamic_ncols=False,
                  desc=pbar_descr_prefix, leave=False) as pbar:

            # TODO: hyper parameters schedulers (Create a class?)

            # -- Start epoch
            for batch_idx, batch in enumerate(dataloader):
                device_batch = self.to_device(batch)

                # -- TODO:
                # 1. log_at_each_epoch

                # -- Model specific schedulers
                self.model.schedulers(self.epoch, logger=logger)

                # -- Execute a training step
                outputs = self.model.training_step(device_batch)

                # -- Save output for each training step
                self.collect_outputs(outputs)

                # -- Eventually log statistics on tensorboard
                if self.model.steps % self.log_every == 0:
                    self.model.log_train(outputs.get('stats', dict()), logger)

                pbar.set_description(
                    pbar_descr_prefix +
                    outputs.get('running_tqdm', ''))
                pbar.update()

            # -- End Epoch
            self.model.reset_train_stats()

            pbar.clear()
            print(pbar_descr_prefix +
                  self.outputs_train[-1][-1].get('final_tqdm', ''))

        return self.outputs_train[-1]

    def to_device(self, tensor_list):
        return to_device(tensor_list, self.device)

    def collect_outputs(self, outputs):
        """
        Collect outputs of training_step for each training epoch
        """

        if outputs is not None and len(outputs.keys()) > 0:
            outputs = detach_dict(outputs)
            self.outputs_train[-1].append(outputs)

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

        pbar_descr_prefix = "    %s - " % log_descr.title()
        # Disable network grad while evaluating the model
        with no_grad_ifnotscript(self.model):
            with tqdm(total=len(dataloader), ncols=80, dynamic_ncols=False,
                      desc=pbar_descr_prefix, leave=False) as pbar:

                for batch_idx, batch in enumerate(dataloader):
                    device_batch = self.to_device(batch)
                    outputs = self.model.validation_step(
                        device_batch, self.epoch)

                    pbar.set_description(
                        pbar_descr_prefix + outputs.get('running_tqdm', ''))
                    pbar.update()

                if logger is not None:
                    self.model.log_val(
                        self.epoch, log_descr,
                        outputs.get('stats', dict()), logger)
                self.model.reset_val_stats()

                pbar.clear()
                print("    %s - %s" % (
                      log_descr.title(), outputs.get('final_tqdm', '')))

        return outputs
