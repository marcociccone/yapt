import os
import sys
import glob
import torch
import logging
import numpy as np

from functools import reduce
from abc import ABC, abstractmethod
from time import gmtime, strftime
from copy import deepcopy

from omegaconf import OmegaConf
from yapt.utils.trainer_utils import detach_dict, to_device
from yapt.utils.utils import safe_mkdirs, is_dict, is_list, get_maybe_missing_args, flatten_dict
from yapt.utils.debugging import IllegalArgumentError

from yapt.loggers.base import LoggerDict
from yapt.loggers.tensorboard import TensorBoardLogger
from yapt.loggers.neptune import NeptuneLogger


def recursive_get(_dict, *keys):
    return reduce(lambda c, k: c.get(k, {}), keys, _dict)


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
    def use_amp(self):
        return self._use_amp

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
                 model_class,
                 extra_args=None,
                 external_logdir=None,
                 init_seeds=True):

        self.console_log = logging.getLogger()

        # -- Load config-arguments from files/dict/cli
        self.load_args(extra_args)
        args = self._args

        self._global_step = 0
        self._verbose = args.verbose
        self._use_cuda = args.cuda and torch.cuda.is_available()
        self._use_amp = False
        self._device = torch.device("cuda" if self._use_cuda else "cpu")
        self.console_log.info("Device: %s", str(self._device))

        # TODO: distributed ?
        self.proc_rank = 0
        self.world_size = 1
        self.node_rank = 0

        # -- Init random seed
        self.init_seeds(init_seeds)
        # -- Logging and Experiment path
        self.log_every = args.loggers.log_every
        self._restart_path = self.get_maybe_missing_args('restart_path')
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
                args.loggers.logdir,
                args.dataset_name.lower(),
                model_class.__name__. lower(),
                self._timestring + "_%s" % args.exp_name)

        self._logger = self.configure_loggers(external_logdir)
        self.dump_args(self._logdir)

        # TODO: here we should handle dataparallel table and distributed mode

        # -- Load Model
        model = model_class(args, logger=self._logger, device=self._device)

        # -- Move model to device
        self._model = model.to(self._device)

        # -- Make a link to trainer to access its properties from model
        self._model.set_trainer(self)

        # -- Print model summary
        if self.proc_rank == 0:
            self.model.summarize()

    def configure_loggers(self, external_logdir=None):
        if external_logdir is not None:
            self._logdir = external_logdir
            self.console_log.warning("external logdir {}".format(
                external_logdir))

        args_logger = self.args.loggers
        loggers = dict()

        # default tensorboard
        # TODO: decide if we can make it optional
        safe_mkdirs(self._logdir, exist_ok=True)
        loggers['tb'] = TensorBoardLogger(self._logdir)

        # Neptune
        if get_maybe_missing_args(args_logger, 'neptune') is not None:
            # TODO: because of api key and sesitive data,
            # neptune project should be per_project in a separate file
            loggers['neptune'] = NeptuneLogger(
                api_key=os.environ['NEPTUNE_API_TOKEN'],
                project_name=args_logger.neptune.project_name,
                offline_mode=args_logger.neptune.offline_mode,
                experiment_name=self.args.exp_name,
                params=flatten_dict(self.args),
                properties=None,
                tags=None,
                logger=self.console_log
            )

        # Wrap loggers
        loggers = LoggerDict(loggers)
        return loggers

    # def prepare_experiment(self):
    #     # link up experiment object
    #     if self.logger is not None:
    #         ref_model.logger = self.logger

    #         # save exp to get started
    #         if hasattr(ref_model, "hparams"):
    #             self.logger.log_hyperparams(ref_model.hparams)

    #         self.logger.save()

    def get_maybe_missing_args(self, key, default=None):
        return get_maybe_missing_args(self.args, key, default)

    # def get_modifiable_args(self, keys=[]):
    #     # This works only on the first level
    #     dict_opt_custom = dict()
    #     dict_opt_extra = dict()
    #     for key in keys:
    #         # -- Save args for later
    #         if not isinstance(key, (list, tuple)):
    #             key = [key]
    #         dict_opt_custom[key]= deepcopy(recursive_get(self._custom_config_args, key))
    #         dict_opt_extra[key] = deepcopy(recursive_get(self._extra_args, key))
    #         # -- Remove from dictionary
    #         if dict_opt_custom is not None:
    #             dc = recursive_get(self._custom_config_args, key)
    #             del dc
    #         if dict_opt_extra is not None:
    #             dc = recursive_get(self._extra_args, key)
    #             del dc

    #     return dict_opt_custom, dict_opt_extra

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

        # -- 0. From command line
        self._cli_args = OmegaConf.from_cli()

        if self._cli_args.config is not None:
            self.default_config = self._cli_args.config
            del self._cli_args['config']
            self.console_log.warning("override default config with: %s", self.default_config)

        # -- 1. From experiment default config file
        self._default_config_args = OmegaConf.load(self.default_config)

        # -- 2. From experiment custom config file (passed from cli)
        self._custom_config_args = OmegaConf.create(dict())
        if self._cli_args.custom_config is not None:
            self._custom_config_args = OmegaConf.load(
                self._cli_args.custom_config)

        # -- 3. Extra config from Tune or any script
        if is_dict(extra_args):
            matching = [s for s in extra_args.keys() if "." in s]
            if len(matching) > 0:
                self.console_log.warning("It seems you are using dotted notation \
                      in a dictionary! Please use a list instead, \
                      to modify the correct values! %s", matching)
            self._extra_args = OmegaConf.create(extra_args)

        elif is_list(extra_args):
            self._extra_args = OmegaConf.from_dotlist(extra_args)

        elif extra_args is None:
            self._extra_args = OmegaConf.create(dict())

        else:
            raise ValueError("extra_args should be a list of \
                             dotted strings or a dict")

        # -- 4. Merge default args
        self._args = OmegaConf.merge(
            self._defaults_yapt,
            self._default_config_args)

        # -- 5. make args structured: it fails if accessing a missing key
        OmegaConf.set_struct(self._args, True)

        # -- Save optimizer args for later
        dict_opt_custom = deepcopy(self._custom_config_args.optimizer)
        dict_opt_extra = deepcopy(self._extra_args.optimizer)
        if dict_opt_custom is not None:
            del self._custom_config_args['optimizer']
        if dict_opt_extra is not None:
            del self._extra_args['optimizer']

        # -- 6. override custom args, ONLY IF THEY EXISTS
        self._args = OmegaConf.merge(
            self._args,
            self._custom_config_args,
            self._extra_args)

        # !!NOTE!! Optimizer could drastically change
        OmegaConf.set_struct(self._args, False)
        if dict_opt_custom is not None:
            self._args = OmegaConf.merge(
                self._args,
                OmegaConf.create({'optimizer': dict_opt_custom}))

        if dict_opt_extra is not None:
            self._args = OmegaConf.merge(
                self._args,
                OmegaConf.create({'optimizer': dict_opt_extra}))
        OmegaConf.set_struct(self._args, True)

        # !!NOTE!! WORKAROUND because of Tune comman line args
        OmegaConf.set_struct(self._args, False)
        self._args = OmegaConf.merge(
            self._args, self._cli_args)
        OmegaConf.set_struct(self._args, True)

    def print_args(self):
        self.console_log.info("Final args:")
        self.console_log.info(self._args.pretty())

        self.console_log.info("Default YAPT args:")
        self.console_log.info(self._defaults_yapt.pretty())

        self.console_log.info("\n\nDefault config args:")
        self.console_log.info(self._default_config_args.pretty())

        self.console_log.info("\n\nCustom config args:")
        self.console_log.info(self._custom_config_args.pretty())

        self.console_log.info("\n\nExtra args:")
        self.console_log.info(self._extra_args.pretty())

        self.console_log.info("\n\ncli args:")
        self.console_log.info(self._cli_args.pretty())

    # def print_verbose(self, message):
    #     if self._verbose:
    #         print(message)

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
            self.console_log.info("Random seed: %d", self._seed)

        if self._use_cuda and self.get_maybe_missing_args('cudnn') is not None:
            torch.backends.cudnn.benchmark = args.cudnn.benchmark
            torch.backends.cudnn.deterministic = args.cudnn.deterministic
            self.console_log.info("cudnn.benchmark: %s", args.cudnn.benchmark)
            self.console_log.info("cudnn.deterministic: %s", args.cudnn.deterministic)

    def set_data_loaders(self):
        raise NotImplementedError("Implement this method to return a dict \
                                   of dataloaders or pass it to the constructor")

    def log_args(self):
        name_str = os.path.basename(sys.argv[0])
        args_str = "".join([("%s: %s, " % (arg, val)) for arg, val in sorted(vars(self._args).items())])[:-2]
        self._logger['tb'].experiment.add_text(
            "Script arguments", name_str + " -> " + args_str)

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
            self.console_log.error("An error occurred while args dump: %s", e)

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
