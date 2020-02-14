import os
import re
import sys
import glob
import json
import torch
import numpy as np

from abc import ABC, abstractmethod
from copy import deepcopy
from time import gmtime, strftime

from yapt.utils.trainer_utils import detach_dict, to_device
from yapt.utils.utils import safe_mkdirs
from yapt.core.logger.tensorboardXsafe import SummaryWriter
from yapt.core.confparser import configparser


class BaseTrainer(ABC):

    default_config = None

    def __init__(self,
                 model_class=None,
                 extra_args=dict(),
                 init_seeds=True):

        self.set_defaults(extra_args)
        args = self.args

        self.init_data = False
        self.verbose = args.trainer.verbose
        self.use_cuda = args.trainer.cuda and torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        self.print_verbose("Device: {}".format(self.device))

        # -- 0. Init random seed
        self.init_seeds(init_seeds)

        # -- X. here it was data loader init and model

        # -- 2. Load Model
        # TODO: here we should handle dataparallel table and distributed mode
        model = self.set_model() if model_class is None \
            else model_class(args, self.device)
        self.model = model.to(self.device)

        # -- Logging and Experiment path
        self.log_every = args.loggers.log_every
        self.restart_path = args.restart_path
        self.timestring = strftime("%Y-%m-%d_%H-%M-%S", gmtime())
        if self.restart_path is not None and not self.args.use_new_dir:
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
                args.loggers.logdir, args.dataset_name.lower(),
                self.timestring + "_%s" % args.exp_name)

        safe_mkdirs(self.logdir, exist_ok=True)
        self.logger = SummaryWriter(log_dir=self.logdir)

    def set_defaults(self, extra_args=dict()):
        # retrieve module path
        dir_path = os.path.dirname(os.path.abspath(__file__))
        dir_path = os.path.split(dir_path)[0]
        # get all the default yaml configs with glob
        dir_path = os.path.join(dir_path, 'configs', '*.yml')

        self.defaults_path = {}
        self.defaults_args = {'defaults': {}}
        for file in glob.glob(dir_path):
            # split filename from path to create key and val
            key = os.path.splitext(os.path.split(file)[1])[0]
            self.defaults_path[key] = file
            # parse default args
            self.defaults_args[key] = configparser._parse_yaml(file)

        if self.defaults_args['defaults'] is not None:
            # defaults.yml contains general args and should not be nested
            self.defaults_args.update(
                deepcopy(self.defaults_args['defaults']))
            del self.defaults_args['defaults']

        self.extra_args = extra_args
        self.args = configparser.parse_configuration(
            self.default_config, dump_config=True,
            external_defaults=self.defaults_args,
            extra_args=extra_args)

    def print_verbose(self, message):
        if self.verbose:
            print(message)

    def init_seeds(self, init_seeds):
        # -- This might be the case the user wants to init the seeds by himself
        # -- For instance, he creates the datasets outside the constructor
        if not init_seeds:
            return

        args = self.args
        self.seed = args.trainer.seed

        if self.seed != -1:
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed(self.seed)
            np.random.seed(self.seed)  # Numpy module.
            # random.seed(self.seed)  # Python random module.
            self.print_verbose("Random seed: {}".format(self.seed))

        if self.use_cuda and args.cudnn is not None:
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

    def log_params(self):
        name_str = os.path.basename(sys.argv[0])
        args_str = "".join([("%s: %s, " % (arg, val)) for arg, val in sorted(vars(self.args).items())])[:-2]
        self.logger.add_text("Script arguments", name_str + " -> " + args_str)

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
            filename = os.path.join(path, filename)

            current_state_dict = {
                'seen': self.seen,
                'epoch': self.epoch,
                'best_epoch': self.best_epoch,
                'beaten_epochs': self.beaten_epochs,
                'best_epoch_score': self.best_epoch_score,
                'model_state_dict': self.model.state_dict(),
            }

            # -- there might be more than one optimizer
            if isinstance(self.model.optimizer, dict):
                optimizer_state_dict = {}
                for key, opt in self.model.optimizer.items():
                    optimizer_state_dict.update({key: opt.state_dict()})
            else:
                optimizer_state_dict = self.model.optimizer.state_dict()

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
        self.seen = checkpoint['seen']
        self.epoch = checkpoint['epoch']
        self.best_epoch = checkpoint['best_epoch']
        self.beaten_epochs = checkpoint['beaten_epochs']
        self.best_epoch_score = checkpoint['best_epoch_score']
        self.model.load_state_dict(checkpoint['model_state_dict'])

        if isinstance(self.model.optimizer, dict):
            for key in self.model.optimizers.keys():
                self.model.optimizers.load_state_dict(
                    checkpoint['optimizer_state_dict'][key])
        else:
            self.model.optimizer.load_state_dict(
                checkpoint['optimizer_state_dict'])

    def to_device(self, tensor_list):
        return to_device(tensor_list, self.device)

    def collect_outputs(self, outputs):
        """
        Collect outputs of training_step for each training epoch
        """
        # TODO: check this, I think it could be generalized
        if outputs is not None and len(outputs.keys()) > 0:
            outputs = detach_dict(outputs)
            self.outputs_train[-1].append(outputs)

    def call_schedulers_optimizers(self):

        schedulers = self.model.scheduler_optimizer

        if isinstance(schedulers, torch.optim.lr_scheduler._LRScheduler):
            schedulers.step()

        elif isinstance(schedulers, dict):
            for _, scheduler in schedulers.items():
                scheduler.step()
        else:
            raise ValueError(
                "optimizers_schedulers should be a \
                dict or a torch.optim.lr_scheduler._LRScheduler object")

    def log_each_epoch(self):
        # -- Log learning rates

        # TODO: maybe a good idea to have a self.optimizers = self.model.optimizer ?
        optimizers = self.model.optimizer
        if isinstance(optimizers, torch.optim.Optimizer):
            current_lr = optimizers.param_groups[0]['lr']
            self.logger.add_scalar('optim/lr', current_lr, self.epoch)

        elif isinstance(optimizers, dict):
            for key, opt in optimizers.items():
                current_lr = opt.param_groups[0]['lr']
                self.logger.add_scalar(
                    'optim/lr_{}'.format(key), current_lr, self.epoch)
        else:
            raise ValueError(
                "optimizer should be a dict or a \
                torch.optim.Optimizer object")

        # -- TODO: add anything else to be logged here

    @abstractmethod
    def _fit(self):
        pass

    def fit(self):
        if self.args.dry_run:
            from pprint import pprint
            print(self.args.dumps.string_yaml)
            pprint(self.extra_args)
        else:
            self._fit()


