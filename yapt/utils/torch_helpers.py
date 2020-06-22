import json
import torch
import numpy as np
import torch.nn as nn
import torch.nn.init as init

from torch import optim
from torch.optim import lr_scheduler
from torch.nn.modules import activation
from collections import OrderedDict
from omegaconf.basecontainer import BaseContainer


def xavier_uniform_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
        init.xavier_uniform_(m.weight)
        if m.bias is not None:
            init.zeros_(m.bias)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        init.ones_(m.weight)
        if m.bias is not None:
            init.zeros_(m.bias)


def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
        init.kaiming_normal_(m.weight)
        if m.bias is not None:
            init.zeros_(m.bias)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        init.ones_(m.weight)
        if m.bias is not None:
            init.zeros_(m.bias)


def normal_init(m, mean, std):
    if isinstance(m, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
        init.normal_(m.weight, mean, std)
        if m.bias is not None:
            init.zeros_(m.bias)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        init.ones_(m.weight)
        if m.bias is not None:
            init.zeros_(m.bias)


def alternate_datasets(labelled, unlabelled):
    datasets = np.array([labelled, unlabelled])
    is_unsup = np.array([False, True])
    len_datasets = np.array([len(d) for d in datasets])
    total_steps = np.sum(len_datasets)

    idx_max = np.argmax(len_datasets)
    idx_min = (idx_max + 1) % 2

    every = 1 + np.ceil(len_datasets[idx_max] / len_datasets[idx_min])
    _count_max = 0
    _count_min = 0
    for idx in range(total_steps):
        if idx % every == 0 or _count_max >= len_datasets[idx_max]:
            _count_min += 1
            yield (next(datasets[idx_min]), is_unsup[idx_min])
        else:
            _count_max += 1
            yield (next(datasets[idx_max]), is_unsup[idx_max])


def to_device(tensor_list, device):
    if isinstance(tensor_list, (list, tuple)):
        new_tensors = []
        for tensor in tensor_list:
            if isinstance(tensor, torch.Tensor):
                new_tensors.append(tensor.to(device))
            else:
                new_tensors.append(to_device(tensor, device))
        return new_tensors

    elif isinstance(tensor_list, dict):
        new_tensors = {}
        for key, val in tensor_list.items():
            new_tensors[key] = to_device(val, device)
        return new_tensors

    elif isinstance(tensor_list, torch.Tensor):
        return tensor_list.to(device)

    else:
        return tensor_list


def detach_tensor(tensor_list, to_numpy=False):
    """
    Helper function to recursively detach a tensor or a list of tensors
    This should be used when storing elements that could accumulate gradients.
    """
    if isinstance(tensor_list, (list, tuple)):
        new_tensors = []
        for tensor in tensor_list:
            if isinstance(tensor, torch.Tensor):
                _new = tensor.detach()
                if to_numpy:
                    _new = _new.numpy()
                new_tensors.append(_new)
            else:
                new_tensors.append(detach_tensor(tensor, to_numpy))

        return new_tensors

    elif isinstance(tensor_list, torch.Tensor):
        _new = tensor_list.detach()
        if to_numpy:
            _new = _new.numpy()
        return _new

    else:
        return tensor_list


def detach_dict(dict_tensor, to_numpy=False):
    """
    Helper function to recursively detach a dictionary of tensors
    This should be used when storing elements that could accumulate gradients.
    """
    for key, val in dict_tensor.items():
        if isinstance(val, dict):
            dict_tensor[key] = detach_dict(val, to_numpy)
        else:
            dict_tensor[key] = detach_tensor(val, to_numpy)

    return dict_tensor


def get_activation(args):

    if isinstance(args, str):
        name = args
        params = {}
    elif isinstance(args, dict, BaseContainer):
        name = args['name']
        params = args.get('params', {})
    else:
        raise ValueError("Only str, dict or OmegaConf are supported")

    activations = OrderedDict({
        'relu': activation.ReLU,
        'relu6': activation.ReLU6,
        'rrelu': activation.RReLU,
        'leaky_relu': activation.LeakyReLU,
        'prelu': activation.PReLU,
        'glu': activation.GLU,
        'elu': activation.ELU,
        'celu': activation.CELU,
        'gelu': activation.GELU,
        'selu': activation.SELU,
        'sigmoid': activation.Sigmoid,
        'tanh': activation.Tanh,
        'hardtanh': activation.Hardtanh,
        'threshold': activation.Threshold,
        'softmax': activation.Softmax,
        'softmax2d': activation.Softmax2d,
        'log_softmax': activation.LogSoftmax,
        'log_sigmoid': activation.LogSigmoid,
        'hardshrink': activation.Hardshrink,
        'softplus': activation.Softplus,
        'softshrink': activation.Softshrink,
        'multihead_attention': activation.MultiheadAttention,
        'softsign': activation.Softsign,
        'softmin': activation.Softmin,
        'tanhshrink': activation.Tanhshrink,
        'identity': nn.Identity,
    })

    from packaging import version
    if version.parse(torch.__version__) >= version.parse("1.5"):
        activations.update({
            'hardsigmoid': activation.Hardsigmoid,
            'hardswitch': activation.Hardswish
        })

    return activations[name.lower()](**params)


def get_optimizer(name):
    optimizers = OrderedDict({
        'adadelta': optim.Adadelta,
        'adagrad': optim.Adagrad,
        'adam': optim.Adam,
        'adamw': optim.AdamW,
        'sparse_adam': optim.SparseAdam,
        'adamax': optim.Adamax,
        'asgd': optim.ASGD,
        'sgd': optim.SGD,
        'rprop': optim.Rprop,
        'msprop': optim.RMSprop,
        'lbfgs': optim.LBFGS
    })
    return optimizers[name.lower()]


def get_scheduler_optimizer(name):
    schedulers = OrderedDict({
        'lambda': lr_scheduler.LambdaLR,
        'step': lr_scheduler.StepLR,
        'multistep': lr_scheduler.MultiStepLR,
        'exponential': lr_scheduler.ExponentialLR,
        'cosine': lr_scheduler.CosineAnnealingLR,
        'reduce_on_plateau': lr_scheduler.ReduceLROnPlateau,
        'cyclic': lr_scheduler.CyclicLR,
        'cosine_warm_restarts': lr_scheduler.CosineAnnealingWarmRestarts,
    })
    return schedulers[name.lower()]


class ToTensor1D(object):
    """Convert a `numpy.ndarray` to tensor. Unlike `ToTensor` from torchvision,
    this converts numpy arrays regardless of the number of dimensions.
    Converts automatically the array to `float32`.
    """
    def __call__(self, array):
        return torch.from_numpy(array.astype('float32'))

    def __repr__(self):
        return self.__class__.__name__ + '()'

class DisableGradNotScriptContext:
    def __init__(self, model):
        self.model = model
        self.script = self.has_scriptmodule(self.all_submodules(model))
        self.context = None

    @staticmethod
    def all_submodules(module):
        submod = list(module.modules())
        for m in module.modules():
            if m != module:
                submod += DisableGradNotScriptContext.all_submodules(m)
        return submod

    @staticmethod
    def has_scriptmodule(module_list):
        for mod in module_list:
            if isinstance(mod, torch.jit.ScriptModule):
                return True
        return False

    def __enter__(self):
        if not self.script:
            self.context = torch.no_grad()
            self.context.__enter__()

    def __exit__(self, *args):
        if not self.script:
            self.context.__exit__(*args)


# Rename class for convenience
no_grad_ifnotscript = DisableGradNotScriptContext


class TensorEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, (
          np.int_, np.intc, np.intp, np.int8,
          np.int16, np.int32, np.int64, np.uint8,
          np.uint16, np.uint32, np.uint64)):
            return int(obj)

        elif isinstance(obj, (
          np.float_, np.float16, np.float32, np.float64)):
            return float(obj)

        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()

        elif isinstance(obj, (torch.Tensor,)):
            return obj.numpy().tolist()

        return json.JSONEncoder.default(self, obj)


# def optimizer_parameters(self, base_lr, params_mult):
#     """
#     Associates network parameters with learning rates
#     :param float base_lr: the basic learning rate
#     :param OrderedDict params_mult: an OrderedDict containing 'param_name':lr_multiplier pairs. All parameters containing
#     'param_name' in their name are be grouped together and assigned to a lr_multiplier*base_lr learning rate.
#     Parameters not matching any 'param_name' are assigned to the base_lr learning_rate
#     :return: A list of dictionaries [{'params': <list_params>, 'lr': lr}, ...]
#     """

#     selected = []
#     grouped_params = []
#     if params_mult is not None:
#         for groupname, multiplier in params_mult.items():
#             group = []
#             for paramname, param in self.model.named_parameters():
#                 if groupname in paramname:
#                     if paramname not in selected:
#                         group.append(param)
#                         selected.append(paramname)
#                     else:
#                         raise RuntimeError("%s matches with multiple parameters groups!" % paramname)
#             if group:
#                 grouped_params.append({'params': group, 'lr': multiplier * base_lr})

#     other_params = [param for paramname, param in self.model.named_parameters() if paramname not in selected]
#     grouped_params.append({'params': other_params, 'lr': base_lr})
#     assert len(selected)+len(other_params) == len(list(self.model.parameters()))

#     return grouped_params
