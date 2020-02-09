from collections import OrderedDict
import json

separator = '----------'


def prettyprint_configuration(parsing_order, config):
    """ Generates a list of (itemname, value) string tuples for later printing
        to both txt file and tensorboard summary.

        Args:
            config: nested dictionary with the configuration

            parsing_order: list of flat `dot notation` paths in the configuration

        Returns:
            The list of (itemname, value) string tuples
    """
    prettyprinted = []
    for item_path in parsing_order:
        prettyprinted.extend(_parse_config_item(item_path, config))

    return prettyprinted


def _parse_config_item(item_path, config):
    """ Parses a flat config path into a (path, value) string tuple.

        Args:
            item_path: the flat path of the item to parse

            config: the configuration dictionary

        Returns:
            The (path, value) string tuple, wrapped in a list for ease of upstream
            handling.
    """
    path = item_path.split('.')
    for p in path:
        if isinstance(config[p], dict):
            config = config[p]
        else:
            if 'losses_specification' in item_path:
                return _prettyprint_loss(item_path, config[p])
            if item_path == 'data.augmentation_pipeline':
                return _prettyprint_augmentation_pipeline(item_path, config[p])
            else:
                return[(item_path, str(config[p]))]

    return [(item_path, separator)]


def _prettyprint_augmentation_pipeline(block_name, pipeline):
    """ Parses the `data.augmentation_pipeline` config block into a list
        of (augmentation, parameters) string tuples.

    Args:
        config: the configuration dictionary

        block_name: the block name ("data.augmentation_pipeline")

        pipeline: the `data.augmentation_pipeline` config block, list of
        [augmentation_name,  {augmentation_params}].

    Returns:
        The list of (augmentation, parameters) string tuples.
    """
    pretty_pipeline = []
    pretty_pipeline.append((block_name, separator))
    for augmentation in pipeline:
        aug_function = block_name + "." + augmentation[0]
        # With default `str` or `repr` methods i'd get a nasty
        # "CommentedMap([('k', v), ...])" representation
        aug_params = json.dumps(augmentation[1])
        pretty_pipeline.append((aug_function, aug_params))

    return pretty_pipeline


def _prettyprint_loss(block_name, target_losses):
    """ Parses the `losses_specification` config block into a list
        of (loss_target.losstype, parameters) string tuples.

        Args:
            block_name: the block name ("losses_specification.losstarget")

            target_losses: the `losses_specification.losstarget` config block, list
            of {loss parameters} for each loss attached to that target.

        Returns:
            The list of (losstarget.losstype, parameters) string tuples
            for the loss target.
    """
    pretty_losses = []
    for loss in target_losses:
        loss_function = block_name + "." + loss['type']
        loss_params = OrderedDict((key, value) for key, value in loss.items() if key != 'type')
        # With default `str` or `repr` methods i'd get a nasty
        # "CommentedMap([('k', v), ...])" representation
        loss_params = json.dumps(loss_params)
        pretty_losses.append((loss_function, loss_params))

    return pretty_losses
