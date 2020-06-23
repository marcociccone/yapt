import os
import random
import glob
import torch
import logging
import numpy as np

from pathlib import Path
from functools import reduce
from abc import ABC, abstractmethod
from time import gmtime, strftime
from copy import deepcopy
from omegaconf import OmegaConf, ListConfig, DictConfig

from yapt.utils.utils import is_dict, is_list, flatten_dict
from yapt.utils.args import get_maybe_missing_args
from yapt.utils.storage import safe_mkdirs
from yapt.utils.torch_helpers import to_device

from yapt.loggers.base import LoggerDict
from yapt.loggers.tensorboard import TensorBoardLogger
from yapt.loggers.neptune import NeptuneLogger


def recursive_get(_dict, *keys):
    return reduce(lambda c, k: c.get(k, {}), keys, _dict)


class BaseTrainer(ABC):
    """

    """

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
    def datadir(self):
        return self._datadir

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
                 init_seeds=True,
                 default_config=None,
                 remap_args_fn=None,
                 mode=None):

        # -- YAPT requires few env variables
        self.check_empty_env_vars()
        self.console_log = logging.getLogger()

        # -- Load config-arguments from files/dict/cli
        if default_config is not None:
            self.default_config = default_config
        self.load_args()
        self.override_with_custom_args(extra_args)

        # -- If restore_path, args are restored from args.yml
        # default_config_args and default_yapt are not used and
        # only extra_args, cli_args and custom_config are considered.
        self._restore_path = self.get_maybe_missing_args('restore_path')
        if self._restore_path == '':
            self._restore_path = None

        if self._restore_path is not None:
            self._args = self.restore_args(self._restore_path)

            # -- Sometimes args change name during development
            # so we need to remap to new names when reloading an old model
            if remap_args_fn is not None and callable(remap_args_fn):
                self._args = remap_args_fn(self._args)

            # -- Override because one could want different args
            # for multiple training stages or at test time
            self.override_with_custom_args(extra_args)

        # -- mode can we train or test. Can be defined from args and init
        self.mode = self._args.mode.lower() if mode is None else mode
        assert self.mode in ('train', 'test'), 'train or test mode only'

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
        self._use_new_dir = self.get_maybe_missing_args('use_new_dir')
        self._timestring = strftime("%Y-%m-%d_%H-%M-%S", gmtime())
        self._datadir = args.data.datadir

        if self._restore_path is not None and not self._use_new_dir:
            if os.path.isfile(self._restore_path):
                self._logdir = os.path.dirname(self._restore_path)
            elif os.path.isdir(self._restore_path):
                self._logdir = self._restore_path
            else:
                raise NotImplementedError(
                    "restore_path %s is not a file nor a dir" %
                    self._restore_path)
        else:
            logdir = args.loggers.logdir
            if args.loggers.debug:
                logdir = os.path.join(logdir, 'debug')

            # TODO: this should be generalized
            self._logdir = os.path.join(
                logdir,
                args.data.dataset_name.lower(),
                model_class.__name__. lower(),
                self._timestring + "_%s" % args.exp_name)
            self.args.loggers.logdir = self._logdir

        # TODO: should restore exp id from neptune
        self._logger = self.configure_loggers(external_logdir)
        self.dump_args(self._logdir)

        # TODO: here we should handle dataparallel table and distributed mode

        # -- Load Model
        model = model_class(args, logger=self._logger, device=self._device)

        # -- Move model to device
        self._model = model.to(self._device)

        # -- Make a link to trainer to access its properties from model
        self._model.set_trainer(self)

    def configure_loggers(self, external_logdir=None):
        """
        YAPT supports logging experiments with multiple loggers at the same time.
        By default, an experiment is logged by TensorBoardLogger.

        external_logdir: if you want to override the logdir.
            It could be useful to use the same directory used by Ray Tune.

        """
        if external_logdir is not None:
            self.args.loggers.logdir = self._logdir = external_logdir
            self.console_log.warning("external logdir {}".format(
                external_logdir))

            if self.args.loggers.debug:
                self.console_log.warning(
                    "Debug flag is disable when external logdir is used")

        # -- No loggers in test mode
        if self.mode == 'test':
            return None

        args_logger = self.args.loggers
        loggers = dict()

        safe_mkdirs(self._logdir, exist_ok=True)

        # -- Tensorboard is not defualt anymore
        if args_logger.tensorboard:
            loggers['tb'] = TensorBoardLogger(self._logdir)

        # -- Neptune
        if (get_maybe_missing_args(args_logger, 'neptune') is not None and
                len(args_logger.neptune.keys()) > 0):
            # TODO: because of api key and sesitive data,
            # neptune project should be per_project in a separate file

            # TODO: THIS THIS SHOULD BE DONE FOR EACH LEAF
            args_neptune = dict()
            for key, val in args_logger.neptune.items():
                if isinstance(val, ListConfig):
                    val = list(val)
                elif isinstance(val, DictConfig):
                    val = dict(val)
                args_neptune[key] = val

            # -- Recursively search for files or extensions
            if 'upload_source_files' in args_neptune.keys():
                source_files = [
                    str(path) for ext in args_neptune['upload_source_files']
                    for path in Path('./').rglob(ext)]
                del args_neptune['upload_source_files']
            else:
                source_files = None

            loggers['neptune'] = NeptuneLogger(
                api_key=os.environ['NEPTUNE_API_TOKEN'],
                experiment_name=self.args.exp_name,
                params=flatten_dict(self.args),
                logger=self.console_log,
                upload_source_files=source_files,
                **(args_neptune))

        # Wrap loggers
        loggers = LoggerDict(loggers)
        return loggers

    def get_maybe_missing_args(self, key, default=None):
        return get_maybe_missing_args(self.args, key, default)

    def check_empty_env_vars(self):
        # list of variables to check are not empty
        env_vars = ['YAPT_LOGDIR', 'YAPT_DATADIR',
                    'NEPTUNE_USER', 'NEPTUNE_API_TOKEN']
        for key in env_vars:
            if os.environ[key] is None:
                self.console_log.warning(
                    "YAPT env variable '{}' is not set".format(key))

    def load_args(self):
        """
        There are several ways to pass arguments via the OmegaConf interface

        In general a Trainer object should have the property `default_config`
        to set the path of the default config file containing all the training
        arguments.

        - `default_config` can be overridden via cli specifying the path
            with the special `config` argument.

        """

        # retrieve module path
        dir_path = os.path.dirname(os.path.abspath(__file__))
        dir_path = os.path.split(dir_path)[0]
        # get all the default yaml configs with glob
        dir_path = os.path.join(dir_path, 'configs', '*.yml')

        # -- From default yapt configuration
        self._defaults_path = {}
        self._defaults_yapt = OmegaConf.create(dict())
        for file in glob.glob(dir_path):
            # split filename from path to create key and val
            key = os.path.splitext(os.path.split(file)[1])[0]
            self._defaults_path[key] = file
            # parse default args
            self._defaults_yapt = OmegaConf.merge(
                self._defaults_yapt, OmegaConf.load(file))

        # -- From command line
        self._cli_args = OmegaConf.from_cli()
        if self._cli_args.config is not None:
            self.default_config = self._cli_args.config
            del self._cli_args['config']
            self.console_log.warning("override default config with: %s", self.default_config)

        # -- From experiment default config file
        self._default_config_args = OmegaConf.create(dict())
        if self.default_config is not None:
            self._default_config_args = OmegaConf.load(self.default_config)

        # -- Merge default args
        self._args = OmegaConf.merge(
            self._defaults_yapt,
            self._default_config_args)

        # -- Resolve interpolations to be sure all nodes are explicit
        self._args = OmegaConf.to_container(self._args, resolve=True)
        self._args = OmegaConf.create(self._args)

        # -- make args structured: it fails if accessing a missing key
        OmegaConf.set_struct(self._args, True)

    def override_with_custom_args(self, extra_args=None):
        """
        Specific arguments can be overridden by:

        - `custom_config` file, defined via cli.
        - `extra_args` dict passed to the constructor of the Trainer object.
        - via command line using the dotted notation.

        The arguments should already defined in the default_config, otherwise
        an exception is raised since you are trying to modify an argument that
        does not exist.
        """

        # -- From command line
        self._cli_args = OmegaConf.from_cli()

        # -- From experiment custom config file (passed from cli)
        self._custom_config_args = OmegaConf.create(dict())
        if self._cli_args.custom_config is not None:
            self._custom_config_args = OmegaConf.load(
                self._cli_args.custom_config)

        # -- Extra config from Tune or any script
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

        # -- Save optimizer args for later
        dict_opt_custom = deepcopy(self._custom_config_args.optimizer)
        dict_opt_extra = deepcopy(self._extra_args.optimizer)
        if dict_opt_custom is not None:
            del self._custom_config_args['optimizer']
        if dict_opt_extra is not None:
            del self._extra_args['optimizer']

        # -- override custom args, ONLY IF THEY EXISTS
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

        # -- Resolve interpolations to be sure all nodes are explicit
        self._args = OmegaConf.to_container(self._args, resolve=True)
        self._args = OmegaConf.create(self._args)

    def restore_args(self, dir):
        """
        Restore dumped args previously saved during a run.
        """
        def path(name):
            return os.path.join(dir, name)

        self.console_log.info("Restore exp args from %s" % path('args.yml'))
        return OmegaConf.load(path('args.yml'))

    def dump_args(self, savedir):
        def path(name):
            if self._restore_path is not None and not self._use_new_dir:
                # - we don't want to overwrite the previous args
                dir = os.path.join(
                    savedir, 'args_restore_%s' % self._timestring)
                safe_mkdirs(dir, exist_ok=True)
                return os.path.join(dir, name)
            else:
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
            self.console_log.error("An error occurred during args dump: %s", e)

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
            torch.cuda.manual_seed_all(self._seed)  # if you are using multi-GPU.
            np.random.seed(self._seed)  # Numpy module.
            random.seed(self._seed)  # Python random module.
            self.console_log.info("Random seed: %d", self._seed)

        if self._use_cuda and self.get_maybe_missing_args('cudnn') is not None:
            torch.backends.cudnn.benchmark = args.cudnn.benchmark
            torch.backends.cudnn.deterministic = args.cudnn.deterministic
            self.console_log.info("cudnn.benchmark: %s", args.cudnn.benchmark)
            self.console_log.info("cudnn.deterministic: %s", args.cudnn.deterministic)

    def set_data_loaders(self):
        raise NotImplementedError("Implement this method to return a dict \
                                   of dataloaders or pass it to the constructor")

    def to_device(self, tensor_list):
        return to_device(tensor_list, self._device)

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
            try:
                self._fit()
            except KeyboardInterrupt:
                self.console_log.info('Detected KeyboardInterrupt, attempting graceful shutdown...')
                self.shutdown()
            self.shutdown()

    def shutdown(self, msg='success'):
        # model = self.get_model()

        if getattr(self, '_train_pbar', None) is not None:
            self._train_pbar.close()
        if getattr(self, '_val_pbar', None) is not None:
            self._val_pbar.close()
        if getattr(self, '_test_pbar', None) is not None:
            self._test_pbar.close()

        # with self.profiler.profile('on_train_end'):
        #     model.on_train_end()

        if self.logger is not None:
            self.logger.finalize(msg)

        # # summarize profile results
        # self.profiler.describe()
