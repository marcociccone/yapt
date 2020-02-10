"""
Universal config specification for training scripts.
"""
import argparse
import itertools
import os
import warnings
import collections.abc

import munch
from ruamel import yaml
# from ruamel.yaml import YAML

from .config_wrapper import ConfigWrapper
from .initializable_parser import InitializableParser


def _parse_yaml(yaml_file):
    """ Opens and parses a YAML file.

        Args:
            yaml_file:
                path to the yaml file

        Returns:
            A `ruamel.yaml.CommentedMap` object containing the parsed yaml file.
    """
    with open(yaml_file, 'r') as yf:
        # yaml = YAML(typ='rt')
        yaml_config = yaml.round_trip_load(yf)

    return yaml_config

# Recursively update external_defaults with default_args
def _update_dict(d, u):
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = _update_dict(d.get(k, {}), v)
        else:
            d[k] = v
    return d

def parse_configuration(default_config, dump_config=True,
                        args_string=None, external_defaults=None, extra_args=None):
    """ Parses a default YAML configuration file, then optionally a custom configuration
        YAML file specified via command line, then finally any other command line arguments.

        Any fields / keys not present in the default configuration, but present
        in either the custom configuration or in the command lines will result in
        errors.
        The command line overrides the custom configuration, which overrides the
        default configuration.
        Use dot notation, e.g. `nesteddictionary.nestedoption` in custom config
        or command line to only override specific nested, non top-level, items.

        Args:
            default_config:
                path to the default configuration file.

            dump_config:
                A 'bool` specifying whether to generate the txt, json and
                yaml config printouts.

            args_string:
                string with args to parse alternatively to sys.argv

            external_defaults:
                A 'dict' containing default values.
                These values could be for instance collected from external
                files or passed from other functions. They can be overrided.

            extra_args:
                A 'dict' of args, this is the last layer of override.
                Mostly helpful when generating custom arguments from code,
                to run tons of experiments.

        Returns:
            A `munch` object containing the final configuration, accessible via
            dot notation.
            https://github.com/Infinidat/munch
            Munch are instances of `dict`, with dot notation and proper support
            of nested dict structures
    """
    # Parse configuration file names
    msg = "\nOverride individual entries of the config with"\
          "\n--entry.subentry.subsubentry value"\
          "\n\n Override boolean config entries by specifying either"\
          "\n--entry.subentry or --no-entry.subentry"\
          "\nSpecify custom configuration path either absolute or relative to root"

    configfiles_parser = argparse.ArgumentParser(usage=msg)

    configfiles_parser.add_argument("--default_config", default=default_config, action="store",
                        help='default configuration file, defines all configuration parameters, must be complete')
    configfiles_parser.add_argument("--custom_config", default=None, action="store",
                        help='custom configuration file, overrides defaults of specified parameters, relative to root')
    config_files, command_line_args = configfiles_parser.parse_known_args(args=args_string)

    default_args = {}
    if default_config is not None:
        config_files.default_config = os.path.abspath(config_files.default_config)
        # Load default configuration
        default_args = _parse_yaml(config_files.default_config)
        # Can pass external defaults as dict
        if external_defaults is not None:
            default_args = _update_dict(external_defaults, default_args)
    else:
        assert external_defaults is not None, \
            'One among `external_defaults` and `default_config` should be specified'

    if config_files.custom_config is not None:
        config_files.custom_config = os.path.abspath(config_files.custom_config)

    config_wrapped = ConfigWrapper(default_args, dump_config)

    # -- Load custom configuration and overrides defaults
    if config_files.custom_config:
        custom_args = _parse_yaml(config_files.custom_config)
        if custom_args:
            config_wrapped.override_with(custom_args)
        else:
            warnings.warn("Specified custom configuration file: {} is empty, skipping."
                          .format(config_files.custom), Warning)

    # Create and populate command line parser with all default args
    cmdline_parser = InitializableParser(config_wrapped)
    cmdline_args = cmdline_parser.parse_args(command_line_args)

    # Overrides with command line args
    config_wrapped.override_with(cmdline_args)

    # Extra args are used to override from dictionary
    # TODO: not sure if should be after or before command line
    if extra_args is not None:
        assert isinstance(extra_args, dict), 'extra_args should be a dict'
        config_wrapped.override_with(extra_args)

    # Finalize and munchify the configuration
    config_wrapped.finalize()
    run_config = munch.munchify(config_wrapped.config)
    return run_config


def flatten_configuration(munch):
    # Retrieve args as list of (k, v) arg pairs
    # Prepend '--' to arg keys and reverts numeric values to strings
    args_tuples = [_parse_argument(key, value) for (key,value) in munch.items()]
    # Flattens list of tuples in single list
    args_flatlist = list(itertools.chain(*args_tuples))
    return args_flatlist


def _parse_argument(key, value):

    if value == True and type(value) == bool:
        return (['--' + key])
    if value == False and type(value) == bool:
        return(['--no-' + key])

    return ('--' + key, str(value))
