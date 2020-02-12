import argparse
from pprint import pprint

class InitializableParser(argparse.ArgumentParser):
    """ A sublclass of `argparse.ArgumentParser`, allowing initialization from
        another structure.
    """
    def __init__(self, wrapped_config):
        """ Constructor.

            Args:
                wrapped_config:
                    A `ConfigWrapper` wrapping the arguments/config
                    to be used to initialize the parser.
        """
        super().__init__()
        for arg_path in wrapped_config.flat_paths:
            arg_value = wrapped_config.read(arg_path)
            if type(arg_value) is not bool:
                self.add_argument("--" + arg_path, default=None, type=type(arg_value))
            else:
                self._add_boolean_argument(arg_path, default=None)
        return

    def parse_args(self, args_string):
        """ Parsing method, overrides the default `argparse.parse_args`.
            Filters all the defined, but not explicitly parsed arguments.

            Args:
                args_string:
                    `string` containing the arguments to be parsed.

            Returns:
                A `Namespace` containing only the arguments which have been
                explicitly parsed.
        """
        parsed_args, unknown = super().parse_known_args(args_string)

        if unknown:
            print("WARNING: unknown command-line arguments")
            pprint(unknown)

        filtered_args = {k: v for k, v in parsed_args.__dict__.items()
                         if v is not None}

        return filtered_args

    def _add_boolean_argument(self, argname, default, helpstring=None, required=False):

        group = self.add_mutually_exclusive_group(required=required)
        group.add_argument("--" + argname, dest=argname, action='store_true', help=helpstring)
        group.add_argument("--no-" + argname, dest=argname, action='store_false', help=helpstring)
        group.set_defaults(**{argname: default})
