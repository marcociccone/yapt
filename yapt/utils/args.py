import os
import neptune

from omegaconf import OmegaConf
from omegaconf.basecontainer import BaseContainer
from inspect import getfullargspec


def get_maybe_missing_args(args, key, default=None):
    if OmegaConf.is_missing(args, key):
        return default
    else:
        return args.get(key)


def reload_args_from_neptune_or_path(args):
    reload_check = (args.from_neptune.exp_id, args.from_path)
    assert sum(bool(el) for el in reload_check) <= 1, \
        "reload.from_neptune or reload.from_path, pick only one!"

    exp_path = None
    exp_args = {}

    if args.from_neptune.exp_id is not None:
        project_name = args.from_neptune.project_name
        exp_id = args.from_neptune.exp_id
        assert project_name is not None, \
            "Specify a project_name to reload a model checkpoint!"
        data = neptune.init(project_name)
        exp = data.get_experiments(id=exp_id)[0]
        params = exp.get_parameters()
        exp_path = params['loggers.logdir']

        # -- reload from yml file
        # its easier to manipulate and access with omegaconf
        exp_args = OmegaConf.load(os.path.join(exp_path, 'args.yml'))
        # checkpoints_dir = os.path.join(exp_path, 'checkpoints')
        print("Reload from Neptune project {} - ID: {} ...".format(
            project_name, exp_id))

    if args.from_path is not None:
        exp_path = args.from_path
        exp_args = OmegaConf.load(os.path.join(exp_path, 'args.yml'))
        # checkpoints_dir = os.path.join(exp_path, 'checkpoints')
        print("Reload from path ...")

    # print("Checkpoints dir: {}".format(checkpoints_dir))
    print("Epoch {}".format(args.epoch))

    # it returns checkpoint_dir and args
    return exp_path, exp_args


class default_args():
    """
        Decorator to load default parameters from file
        and override them
    """
    def __init__(self, defaults, path=None):
        # get args from default_args
        self.default_args = OmegaConf.load(defaults)
        if path is not None:
            for level in path.split("."):
                self.default_args = self.default_args[level]
        # avoid setting args that do not exist in defaults
        OmegaConf.set_struct(self.default_args, True)

        self.error_message = (
            "You can ovverride default_args defined in {}/{}! "
            "To use default_args use args=None!".format(defaults, path)
        )

    def __call__(self, fn):
        """
            The decorator is based on the assumption that the structure
            of the argument is fixed.

            fn(arg_1, ... arg_n, omegaconf_arg=None)

            default args are taken from the *.yml source and are overridden
            if specified.
        """

        def replace(arg):
            if arg is None:
                return self.default_args

            if isinstance(arg, (BaseContainer, dict)):
                # -- This will raise an error if some args
                # are not defined in self.default_args
                resolved_arg = OmegaConf.to_container(
                    OmegaConf.create(arg), resolve=True)
                return OmegaConf.merge(self.default_args, resolved_arg)
            else:
                raise ValueError(self.error_message)

        def wrapped_f(*args, **kwargs):
            if 'args' in kwargs.keys():
                kwargs['args'] = replace(kwargs['args'])
            else:
                # get index of  omegaconf args in *args
                argspec = getfullargspec(fn)
                index = argspec.args.index('args')

                args = list(args)
                if len(args) - 1 < index:
                    kwargs['args'] = self.default_args
                else:
                    if args[index] is None:
                        args[index] = self.default_args
                    else:
                        args[index] = replace(args[index])

            fn(*args, **kwargs)
        return wrapped_f
