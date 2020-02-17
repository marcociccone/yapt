import os
import sys
import glob
import torch
import numpy as np

from abc import ABC, abstractmethod
from time import gmtime, strftime

from omegaconf import OmegaConf
from yapt.utils.trainer_utils import detach_dict, to_device
from yapt.utils.utils import safe_mkdirs, is_dict, is_list
from yapt.core.logger.tensorboardXsafe import SummaryWriter


def get_maybe_missing(args, key, default=None):
    if OmegaConf.is_missing(args, key):
        return default
    else:
        return args.get(key)


class BaseTrainer(ABC):

    default_config = None

    @property
    def seed(self):
        return self._seed

    @property
    def model(self):
        return self._model

    @property
    def device(self):
        return self._device

    @property
    def logger(self):
        return self._logger

    @property
    def logdir(self):
        return self._logdir

    @property
    def timestring(self):
        return self._timestring

    @property
    def use_cuda(self):
        return self._use_cuda

    @property
    def global_step(self):
        return self._global_step

    @property
    def args(self):
        return self._args

    @property
    def defaults_yapt(self):
        return self._defaults_yapt

    @property
    def default_config_args(self):
        return self._default_config_args

    @property
    def custom_config_args(self):
        return self._custom_config_args

    @property
    def extra_args(self):
        return self._extra_args

    @property
    def cli_args(self):
        return self._cli_args

    def __init__(self,
                 model_class=None,
                 extra_args=None,
                 external_logdir=None,
                 init_seeds=True):

        self.load_args(extra_args)
        args = self._args

        self._global_step = 0
        self._verbose = args.verbose
        self._use_cuda = args.cuda and torch.cuda.is_available()
        self._device = torch.device("cuda" if self._use_cuda else "cpu")
        self.print_verbose("Device: {}".format(self._device))

        # -- 0. Init random seed
        self.init_seeds(init_seeds)

        # -- X. here it was data loader init and model

        # -- Logging and Experiment path
        self.log_every = args.loggers.log_every
        self._restart_path = get_maybe_missing(args, 'restart_path')
        self._timestring = strftime("%Y-%m-%d_%H-%M-%S", gmtime())
        if self._restart_path is not None and not self._args.use_new_dir:
            if os.path.isfile(self._restart_path):
                self._logdir = os.path.dirname(self._restart_path)
            elif os.path.isdir(self._restart_path):
                self._logdir = self._restart_path
            else:
                raise NotImplementedError(
                    "Restart Path %s is not a file nor a dir" %
                    self._restart_path)
        else:
            self._logdir = os.path.join(
                args.loggers.logdir, args.dataset_name.lower(),
                self._timestring + "_%s" % args.exp_name)

        self._logger = self.create_logger(external_logdir)
        self.dump_args(self._logdir)

        # -- 2. Load Model
        # TODO: here we should handle dataparallel table and distributed mode
        model = self.set_model() if model_class is None \
            else model_class(args, logger=self._logger, device=self._device)
        self._model = model.to(self._device)

    def create_logger(self, external_logdir=None):
        if external_logdir is not None:
            self._logdir = external_logdir
            self.print_verbose("WARNING: external logdir {}".format(
                external_logdir))

        safe_mkdirs(self._logdir, exist_ok=True)
        return SummaryWriter(log_dir=self._logdir)

    def load_args(self, extra_args=None):
        # retrieve module path
        dir_path = os.path.dirname(os.path.abspath(__file__))
        dir_path = os.path.split(dir_path)[0]
        # get all the default yaml configs with glob
        dir_path = os.path.join(dir_path, 'configs', '*.yml')

        # -- 0. From default yapt configuration
        self._defaults_path = {}
        self._defaults_yapt = OmegaConf.create(dict())
        for file in glob.glob(dir_path):
            # split filename from path to create key and val
            key = os.path.splitext(os.path.split(file)[1])[0]
            self._defaults_path[key] = file
            # parse default args
            self._defaults_yapt = OmegaConf.merge(
                self._defaults_yapt, OmegaConf.load(file))

        # -- 1. From experiment default config file
        self._default_config_args = OmegaConf.load(self.default_config)

        # -- 2. From command line
        self._cli_args = OmegaConf.from_cli()

        # -- 3. From experiment custom config file (passed from cli)
        self._custom_config_args = OmegaConf.create(dict())
        if self._cli_args.custom_config is not None:
            self._custom_config_args = OmegaConf.load(
                self._cli_args.custom_config)

        # -- 4. Extra config from Tune or any script
        if is_dict(extra_args):
            matching = [s for s in extra_args.keys() if "." in s]
            if len(matching) > 0:
                print("WARNING: it seems you are using dotted notation \
                      in a dictionary! Please use a list instead, \
                      to modify the correct values!")
                print(matching)
            self._extra_args = OmegaConf.create(extra_args)

        elif is_list(extra_args):
            self._extra_args = OmegaConf.from_dotlist(extra_args)

        elif extra_args is None:
            self._extra_args = OmegaConf.create(dict())

        else:
            raise ValueError("extra_args should be a list of \
                             dotted strings or a dict")

        # -- 5. Merge defautl args
        self._args = OmegaConf.merge(
            self._defaults_yapt,
            self._default_config_args)
        # -- 6. make args structured: it fails if accessing a missing key
        OmegaConf.set_struct(self._args, True)
        # -- 7. override custom args, ONLY IF THEY EXISTS
        self._args = OmegaConf.merge(
            self._args,
            self._custom_config_args,
            self._extra_args)

        # !!NOTE!! WORKAROUND because of Tune comman line args
        OmegaConf.set_struct(self._args, False)
        self._args = OmegaConf.merge(
            self._args, self._cli_args)
        OmegaConf.set_struct(self._args, True)

    def print_args(self):
        print("Final args:")
        print(self._args.pretty())

        print("Default YAPT args:")
        print(self._defaults_yapt.pretty())

        print("\n\nDefault config args:")
        print(self._default_config_args.pretty())

        print("\n\nCustom config args:")
        print(self._custom_config_args.pretty())

        print("\n\nExtra args:")
        print(self._extra_args.pretty())

        print("\n\ncli args:")
        print(self._cli_args.pretty())

    def print_verbose(self, message):
        if self._verbose:
            print(message)

    def init_seeds(self, init_seeds):
        # -- This might be the case the user wants to init the seeds by himself
        # -- For instance, he creates the datasets outside the constructor
        if not init_seeds:
            return

        args = self._args
        self._seed = args.seed

        if self._seed != -1:
            torch.manual_seed(self._seed)
            torch.cuda.manual_seed(self._seed)
            np.random.seed(self._seed)  # Numpy module.
            # random.seed(self.seed)  # Python random module.
            self.print_verbose("Random seed: {}".format(self._seed))

        if self._use_cuda and args.cudnn is not None:
            torch.backends.cudnn.benchmark = args.cudnn.benchmark
            torch.backends.cudnn.deterministic = args.cudnn.deterministic
            self.print_verbose("cudnn.benchmark: {}".format(args.cudnn.benchmark))
            self.print_verbose("cudnn.deterministic: {}".format(args.cudnn.deterministic))
        self.print_verbose("")

    def set_data_loaders(self):
        raise NotImplementedError("Implement this method to return a dict \
                                   of dataloaders or pass it to the constructor")

    def set_model(self):
        raise NotImplementedError("Implement this method to return your model \
                                   or pass it to the constructor")

    def log_args(self):
        name_str = os.path.basename(sys.argv[0])
        args_str = "".join([("%s: %s, " % (arg, val)) for arg, val in sorted(vars(self._args).items())])[:-2]
        self._logger.add_text("Script arguments", name_str + " -> " + args_str)

    def dump_args(self, savedir):
        def path(name):
            return os.path.join(savedir, name)
        try:
            self._args.save(path('args.yml'))
            # -- Just to be sure, but not really useful dumps
            self._defaults_yapt.save(path('defaults_yapt.yml'))
            self._cli_args.save(path('cli_args.yml'))
            self._extra_args.save(path('extra_args.yml'))
            self._default_config_args.save(path('default_config_args.yml'))
            self._custom_config_args.save(path('custom_config_args.yml'))

        except Exception as e:
            print("An error occurred while args dump:")
            print(e)

    def log_gradients(self, logger, global_step):
        for name, param in self._model.named_parameters():
            if param.requires_grad and param.grad is not None:
                logger.add_scalar("gradients/" + name, param.grad.norm(2).item(),
                                  global_step=global_step)

    def to_device(self, tensor_list):
        return to_device(tensor_list, self._device)

    def collect_outputs(self, outputs):
        """
        Collect outputs of training_step for each training epoch
        """
        # TODO: check this, I think it could be generalized
        if outputs is not None and len(outputs.keys()) > 0:
            outputs = detach_dict(outputs)
            self.outputs_train[-1].append(outputs)

    def call_schedulers_optimizers(self):

        schedulers = self._model.scheduler_optimizer

        if isinstance(schedulers, torch.optim.lr_scheduler._LRScheduler):
            schedulers.step()

        elif is_dict(schedulers):
            for _, scheduler in schedulers.items():
                scheduler.step()
        else:
            raise ValueError(
                "optimizers_schedulers should be a \
                dict or a torch.optim.lr_scheduler._LRScheduler object")

    @abstractmethod
    def _fit(self):
        pass

    def fit(self):
        if self._args.dry_run:
            self.print_args()
        else:
            self._fit()
