from copy import deepcopy
import json

from numpy import iinfo
from numpy.random import randint
from ruamel import yaml
from ruamel.yaml.compat import StringIO

from .prettyprint import prettyprint_configuration

class ConfigWrapper:
    """ Wraps a `dictionary`, or derivative class, providing an interface
        to read/write to items in nested dictionaries via dot notation, to
        override items with matching keys/paths and types from another dictionary,
        and to perform finalization operations before the configuration will be
        run.

        NO MODIFICATIONS TO THE CONFIG OBJECT ARE ACCEPTABLE OUTSIDE THIS
        OBJECT.

        TODO: Maybe turn this into a child class of munch
    """

    def __init__(self, dictionary, dump=False):
        """ Constructor.

            Args:
                dictionary:
                    A dictionary or dictionary derived class.

                dump:
                    A 'bool` specifying whether to generate the txt, json and
                    yaml config printouts.
        """
        if isinstance(dictionary, dict):
            self.flat_paths = self._resolve_dict_paths(dictionary)
            self.config = dictionary
            self.dump = dump
        else:
            raise TypeError("Configuration object is not an instance of `dict`")

    def _resolve_dict_paths(self, d, parent=''):
        """ Parses a dictionary and returns the `paths` to items, nested in
            sub dictionaries or not, as dot notation,
            e.g. d[subdict][item] -> subdict.item

            Args:
                d:
                    top-level dictionary.

                parent:
                    `string`, parent path from the top of `d`.

            Returns:
                The list of flat item paths for the top-level dictionary.
        """
        flat_paths = []
        if isinstance(d, dict):
            for key in d:
                path = parent + '.' + key if parent != '' else key
                flat_paths.append(path)
                if isinstance(d[key], dict):
                    flat_paths.extend(self._resolve_dict_paths(d[key], path))
        return flat_paths

    def read(self, flatpath):
        """ Reads the value in the position specified by flatpath.

            Args:
                flatpath:
                    A string in the format, e.g. `subdict.subsubdict.item`.

            Returns:
                The value at self.hieerarchical[subdict][subsubdict][item].
        """
        element = self.config
        split_path = flatpath.split('.')
        for path_level in range(len(split_path)):
            try:
                element = element[split_path[path_level]]
            except KeyError:
                # Maybe we don't have all nesting levels actually in the config
                # e.g. with the launcher configs
                partial_path = '.'.join(split_path[path_level:])
                element = element[partial_path]
                break
        return element

    def write(self, flatpath, value):
        """ Writes `value` in the position specified by flatpath.

            Args:
                flatpath:
                    A string in the format, e.g. `subdict.subsubdict.item`.

                value:
                    The value to be written at `flatpath`.
        """
        element = self.config
        split_path = flatpath.split('.')
        last_key = split_path[-1]
        for path_level in range(len(split_path) - 1):
            try:
                element = element[split_path[path_level]]
            except KeyError:
                # Maybe we don't have all nesting levels actually in the config
                # e.g. with the launcher configs
                last_key = '.'.join(split_path[path_level:])
                break
        try:
            element[last_key] = value
        except KeyError as e:
            err_msg = ("Configuration item \"{}\" not found".format(flatpath))
            raise e(err_msg)

    def override_with(self, override_config):
        """ Overrides values in `self.hierarchical' with those in override_config.

            Args:
                override_config:
                    A dictionary with the (key, value) pairs of items
                    whose values in `self.config` are to be overridden.
        """
        for arg_key, arg_value in override_config.items():
            if arg_key in self.flat_paths:

                try:
                    arg_type = type(self.read(arg_key))
                    if not isinstance(arg_value, arg_type):
                        err_msg = "Type mismatch between baseline and override element {}: \
                              expected {}, got {}".format(arg_key, arg_type.__name__, type(arg_value).__name__)
                        raise TypeError(err_msg)
                    self.write(arg_key, arg_type(arg_value))

                except TypeError as e:
                    err_msg = "Type conversion error between baseline and override element '{}'".format(arg_key)
                    raise e(err_msg)
            else:
                err_msg = "Argument '{}' in overriding configuration does not have a matching baseline element".format(arg_key)
                raise KeyError(err_msg)

            # Update flatpaths with overridden configuration
            # (We might be redefining top level blocks whose structure was overridden
            # in a different way, e.g. loss targets, augmentation pipeline, etc...)
            # TODO: review when data structures will be updated
            self.flat_paths = self._resolve_dict_paths(self.config)

    def finalize(self):
        """ Finalizes the configuration.
        """
        self._finalize_seed()
        if self.dump:
            self._generate_dumps()

    def _finalize_seed(self):
        """ Initializes the seed(s) argument with a random value if not explicitly
            specified. Runs on every config argument including `seed` in its leaf name.
            e.g. 'config.seed', 'config.output.trainval_seed'
        """
        for config_value in self.flat_paths:
            path = config_value.split('.')
            if 'seed' in path[-1]:
                seed_value = self.read(config_value)
                if seed_value == -1:
                    self.write(config_value, randint(iinfo('uint32').max))

    def _generate_json(self, config):
        """ Generates a JSON string for the configuration.

            Args:
                config: the configuration
            Returns:
                The JSON string for the configuration.
        """
        json_string = json.dumps(config, indent=4)
        return json_string

    def _generate_yaml(self, config):
        """ Generates a YAML string for the configuration.

        Args:
            config:
                The configuration
        Returns:
            The YAML string for the configuration. Includes all comments found
            in the default configuration too.
        """
        # yaml = YAML(typ='rt')
        yaml_out_stream = StringIO()
        yaml.round_trip_dump(config, yaml_out_stream, indent=4)
        yaml_out_string = yaml_out_stream.getvalue()
        return yaml_out_string

    def _generate_tuples(self, config):
        """ Generates flat (itemname, value) tuples for the configuration.

        Args:
            config:
                The configuration
        Returns:
            A list of (itemname, value) tuples for the configuration.
        """
        string_tuples = prettyprint_configuration(self.flat_paths, config)
        return string_tuples

    def _generate_dumps(self):
        """ Generates flat, JSON, and YAML representations of `self.config` and
            attaches them to the relevant fields for ease of transport.
        """
        # Copying the configuration so not to have the serialized representations
        # inside the other serialized representations.
        clean_config = deepcopy(self.config)
        self.config['dumps'] = {}
        self.config['dumps']['string_json'] = self._generate_json(clean_config)
        self.config['dumps']['string_yaml'] = self._generate_yaml(clean_config)
        self.config['dumps']['string_tuples'] = self._generate_tuples(clean_config)


#   EXPERIMENTAL: improved mechanism for flat / nested path resolution
#    def _normalize_hive(self, dictionary, flatpaths):
#
#        for fp in flatpaths:
#            d = dictionary
#            split_path = fp.split('.')
#            l = len(split_path) - 1
#            for i in range(l):
#                cur = split_path[0]
#                split_path = split_path[1:]
#                current_key = '.'.join([cur] + split_path)
#                normalized_key = '.'.join(split_path)
#
#                if cur not in d.keys():
#                    d[cur] = {}
#                d_next = d[cur]
#                if normalized_key not in d_next.keys():
#                    d_next[normalized_key] = d[current_key]
#                    del d[current_key]
#                d = d_next
#
#        return dictionary
