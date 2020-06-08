import torch
import inspect
import hashlib
import os
import json
import itertools
import logging
import numpy as np
import matplotlib.pyplot as plt

from functools import wraps
from time import time
from PIL import Image
from textwrap import wrap
from omegaconf import ListConfig, DictConfig
from collections import MutableMapping
from omegaconf import OmegaConf

log = logging.getLogger(__name__)


def recursive_keys(_dict):
    keys = []
    for k, v in _dict.items():
        if isinstance(v, dict):
            keys += [{k: recursive_keys(v)}]
        else:
            keys += [k]
    return keys


def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print('%r took: %2.4f sec' %  (f.__name__, te-ts))
        return result
    return wrap


def is_scalar(x):
    if isinstance(x, (float, int)):
        return True
    elif (torch.is_tensor(x) and x.ndim == 0):
        return True
    elif (isinstance(x, np.ndarray) and x.ndim == 0):
        return True
    else:
        return False


def add_key_dict_prefix(_dict, prefix, sep='/'):
    # -- add prefix for each key
    old_keys = list(_dict.keys())
    for key in old_keys:
        new_key = "{}{}{}".format(prefix, sep, key)
        _dict[new_key] = _dict.pop(key)
    return _dict


def get_maybe_missing_args(args, key, default=None):
    if OmegaConf.is_missing(args, key):
        return default
    else:
        return args.get(key)


def is_list(obj):
    return isinstance(obj, (list, tuple, ListConfig))


def is_dict(obj):
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


def call_counter(func):
    def helper(*args, **kwargs):
        helper.calls += 1
        return func(*args, **kwargs)
    helper.calls = 0
    helper.__name__ = func.__name__

    return helper


def warning_not_implemented(console_log=None, level=1):
    # on first index:
    # - 0 is for the called function
    # - 1 is for the caller
    if console_log is None:
        console_log = log
    name = inspect.stack()[level][3]
    console_log.warning("%s method not implemented", name)


def flatten_dict(d, parent_key='', sep='.'):
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


def is_jsonable(x):
    try:
        json.dumps(x)
        return True
    except:
        return False


def clean_unjsonable_keys(arg_dict):
    keys_to_delete = []
    for key, value in arg_dict.items():
        if not is_jsonable(value):
            keys_to_delete.append(key)

    new_dict = dict()
    for kk, vv in arg_dict.items():
        if kk not in keys_to_delete:
            new_dict[kk] = vv
    return new_dict


def save_options(basedir, argparse_opt, sacred_opt=None):

    argparse_dict = clean_unjsonable_keys(vars(argparse_opt))
    with open('%s/opt.json' % basedir, 'w') as outfile:
        json.dump(argparse_dict, outfile)

    if sacred_opt is not None:
        sacred_dict = clean_unjsonable_keys(sacred_opt)
        # Store sacred_opt dictionary
        with open('%s/sacred_cfg.json' % basedir, 'w') as outfile:
            json.dump(sacred_dict, outfile)


def load_options(basedir):
    argparse_opt, sacred_opt = None, None
    argparse_filename = '%s/opt.json' % basedir
    if os.path.isfile(argparse_filename):
        with open(argparse_filename, 'r') as f:
            argparse_opt = json.load(f)

    sacred_filename = '%s/sacred_cfg.json' % basedir
    if os.path.isfile(sacred_filename):
        with open(sacred_filename, 'r') as f:
            sacred_opt = json.load(f)

    return argparse_opt, sacred_opt


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """Saves checkpoint to disk"""

    for idx in range(10):
        try:
            torch.save(state, filename)
            break
        except IOError:
            log.error("IOError: couldn't save the checkpoint after 10 trials")

    filenames = {'last': filename}
    if is_best:
        filename_best = os.path.join(
            os.path.dirname(filename), 'best_model.weights')

        for idx in range(10):
            try:
                torch.save(state, filename_best)
                break
            except IOError:
                log.error("IOError: couldn't save the checkpoint after 10 trials")

        filenames['best'] = filename_best
    return filenames


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


def make_rows_for_comparison(x, x_rec, nrows=4):
    assert (x.shape == x_rec.shape), \
        "x and x_rec should have the same shape"

    comparisons = []
    img_per_row = 8
    n = min(x.size(0), img_per_row)
    for row in range(nrows):
        end = min((row + 1) * n, x.size(0))

        # -- Only get n images
        original_imgs = x[row * n: end]
        rec_imgs = x_rec[row * n: end]
        comparisons.append(original_imgs)
        comparisons.append(rec_imgs)

    comparisons = torch.cat(comparisons, dim=0)
    comparisons = comparisons.expand(-1, 3, -1, -1)

    return comparisons

# Converts a Tensor into an image array (numpy)
# |imtype|: the desired type of the converted numpy array
def tensor2im(input_image, imtype=np.uint8):
    if isinstance(input_image, torch.Tensor):
        image_tensor = input_image.data
    else:
        return input_image
    image_numpy = image_tensor[0].cpu().float().numpy()
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    return image_numpy.astype(imtype)


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


def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)


def print_numpy(x, val=True, shp=False):
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def safe_mkdirs(path, exist_ok=True):
    import errno
    try:
        # Use the same directory as the restart experiment
        os.makedirs(path, exist_ok=exist_ok)
    except Exception as err:
        # get the name attribute from the exception class
        if (type(err).__name__ == 'OSError' and
                err.errno == errno.ENAMETOOLONG):
            # handle specific to OSError [Errno 36]
            os.makedirs(path[:143], exist_ok=exist_ok)
        else:
            raise  # if you want to re-raise; otherwise code your ignore


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def confusion_matrix_fig(cm, labels, normalize=False):

    if normalize:
        cm = cm.astype('float') * 10 / cm.sum(axis=1)[:, np.newaxis]
        cm = np.nan_to_num(cm, copy=True)
        cm = cm.astype('int')

    fig = plt.figure(figsize=(7, 7), facecolor='w', edgecolor='k')
    ax = fig.add_subplot(1, 1, 1)
    im = ax.imshow(cm, cmap='Oranges')

    classes = ['\n'.join(wrap(l, 40)) for l in labels]

    tick_marks = np.arange(len(classes))

    ax.set_xlabel('Predicted', fontsize=7)
    ax.set_xticks(tick_marks)
    c = ax.set_xticklabels(classes, fontsize=4, rotation=-90, ha='center')
    ax.xaxis.set_label_position('bottom')
    ax.xaxis.tick_bottom()

    ax.set_ylabel('True Label', fontsize=7)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes, fontsize=4, va='center')
    ax.yaxis.set_label_position('left')
    ax.yaxis.tick_left()

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, format(cm[i, j], 'd') if cm[i, j] != 0 else '.',
                horizontalalignment="center", fontsize=6,
                verticalalignment='center', color="black")

    return fig

class LinearScheduler:

    def __init__(self, init_step, final_step, init_value, final_value):
        assert final_step >= init_step
        self.init_step = init_step
        self.final_step = final_step
        self.init_value = init_value
        self.final_value = final_value

    def get_value(self, step):

        if step < self.init_step:
            return self.init_value
        elif step >= self.final_step:
            return self.final_value
        else:
            if self.init_step == self.final_step:
                return self.final_value

            rate = (float(self.final_value - self.init_value) /
                    float(self.final_step - self.init_step))
            return self.init_value + rate * (step - self.init_step)

