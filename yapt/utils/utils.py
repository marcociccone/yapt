import torch
import inspect
import hashlib
import json
import logging
import numpy as np

log = logging.getLogger(__name__)


def recursive_keys(_dict):
    """
    Helper function that visits recursively a dictionary and print its keys.
    It is useful for debugging and exception messages.
    """
    keys = []
    for k, v in _dict.items():
        if isinstance(v, dict):
            keys += [{k: recursive_keys(v)}]
        else:
            keys += [k]
    return keys


def listdict_to_dictlist(list_dict):
    return{k: [dic[k] for dic in list_dict] for k in list_dict[0]}


def add_key_dict_prefix(_dict, prefix, sep='/'):
    # -- add prefix for each key
    old_keys = list(_dict.keys())
    for key in old_keys:
        new_key = "{}{}{}".format(prefix, sep, key)
        _dict[new_key] = _dict.pop(key)
    return _dict


def is_scalar(x):
    if isinstance(x, (float, int)):
        return True
    elif (torch.is_tensor(x) and x.ndim == 0):
        return True
    elif (isinstance(x, np.ndarray) and x.ndim == 0):
        return True
    else:
        return False


def is_list(obj):
    from omegaconf import ListConfig
    return isinstance(obj, (list, tuple, ListConfig))


def is_dict(obj):
    from omegaconf import DictConfig
    return isinstance(obj, (dict, DictConfig))


def is_optimizer(obj):
    return isinstance(obj, torch.optim.Optimizer)


def is_dataset(obj):
    return isinstance(obj, torch.utils.data.Dataset)


def is_notebook():
    try:
        from IPython import get_ipython
        shell = get_ipython().__class__.__name__
        module = get_ipython().__class__.__module__

        if shell == 'ZMQInteractiveShell' or module == "google.colab._shell":
            return True   # Jupyter notebook, colab or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)

    except NameError:
        return False      # Probably standard Python interpreter


def stats_to_str(stats, fmt=":.4f"):
    assert isinstance(stats, dict), \
        "stats should be a dict instead is a {}".format(type(stats))
    string = ''
    for key, val in stats.items():
        if torch.is_tensor(val) and val.ndim == 0:
            val = val.item()
            string += ("{}: {" + fmt + "} - ").format(key, val)
    return string


def warning_not_implemented(console_log=None, level=1):
    # on first index:
    # - 0 is for the called function
    # - 1 is for the caller
    if console_log is None:
        console_log = log
    name = inspect.stack()[level][3]
    console_log.warning("%s method not implemented", name)


def flatten_dict(d, parent_key='', sep='.'):
    from collections import MutableMapping
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def make_hash(o_dict):
    d = hashlib.sha1(json.dumps(o_dict, sort_keys=True).encode())
    return d.hexdigest()


def hash_from_time(lenght=10):
    from time import time
    hash = hashlib.sha1()
    hash.update(str(time()).encode('utf-8'))
    return hash.hexdigest()[:lenght]


def reshape_parameters(named_parameters, permutation=(3, 2, 1, 0)):
    parameters = {}
    for name, p in named_parameters:
        if len(p.shape) == 4:
            pp = p.permute(permutation).data.cpu().numpy()
        else:
            pp = p.data.cpu().numpy()
        parameters[name] = pp
    return parameters


def reshape_activations(outputs, permutation=(0, 2, 3, 1)):
    activations = {}
    for name, act in outputs.items():
        if len(act.shape) == 4:
            aa = act.data.permute(permutation).cpu().numpy()
        else:
            aa = act.data.cpu().numpy()
        activations[name] = aa
    return activations


def clamp(value, min_value, max_value):
    return max(min_value, min(value, max_value))


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res, correct


def diagnose_network(net, name='network'):
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def print_numpy(x, val=True, shp=False):
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))
