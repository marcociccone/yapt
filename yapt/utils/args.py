from omegaconf import OmegaConf
from omegaconf.basecontainer import BaseContainer
from inspect import getfullargspec


def get_maybe_missing_args(args, key, default=None):
    if OmegaConf.is_missing(args, key):
        return default
    else:
        return args.get(key)


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
                return OmegaConf.merge(self.default_args, arg)
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
