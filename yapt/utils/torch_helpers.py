import json
import torch
import numpy as np
import torch.nn as nn
import torch.nn.init as init

from functools import partial
from typing import Sequence
from torch import optim
from torch.optim import lr_scheduler
from torch.nn.modules import activation
from collections import OrderedDict
from omegaconf.basecontainer import BaseContainer


linear_layers = (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d,
                 nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)

norm_layers = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
               nn.GroupNorm, nn.LayerNorm)


def calculate_gain(nonlinearity):
    nonlinearity = 'linear' if nonlinearity == 'identity' else nonlinearity
    gain_nonlinearities = [
        'conv1d', 'conv2d', 'conv3d',
        'conv_transpose1d', 'conv_transpose2d', 'conv_transpose3d',
        'linear', 'tanh', 'sigmoid', 'relu', 'leaky_relu'
    ]
    if nonlinearity in gain_nonlinearities:
        return init.calculate_gain(nonlinearity)
    else:
        return 1.0

def get_init_linears_fn(name, nonlinearity='relu', bias_init=0.,
                        **init_fn_params):

    inits = OrderedDict({
        'xavier_uniform': xavier_uniform_init,
        'xavier_normal': xavier_normal_init,
        'kaiming_uniform': kaiming_uniform_init,
        'kaiming_normal': kaiming_normal_init
    })

    init_fn = partial(inits[name], apply_to=linear_layers,
                      nonlinearity=nonlinearity, bias_init=bias_init,
                      **init_fn_params)
    return init_fn


def norm_layers_init(m, gamma_init=1.0, bias_init=0.):
    if isinstance(m, norm_layers):
        if m.weight is not None:
            init.constant_(m.weight, gamma_init)
        if m.bias is not None:
            init.constant_(m.bias, bias_init)


def xavier_uniform_init(m, apply_to=linear_layers,
                        nonlinearity='relu', bias_init=0.,
                        **init_fn_params):
    gain = calculate_gain(nonlinearity)
    if isinstance(m, apply_to):
        init.xavier_uniform_(m.weight, gain=gain, **init_fn_params)
        if m.bias is not None:
            init.constant_(m.bias, bias_init)


def xavier_normal_init(m, apply_to=linear_layers,
                       nonlinearity='relu', bias_init=0.,
                       **init_fn_params):

    gain = calculate_gain(nonlinearity)
    if isinstance(m, apply_to):
        init.xavier_normal_(m.weight, gain=gain, **init_fn_params)
        if m.bias is not None:
            init.constant_(m.bias, bias_init)


def kaiming_uniform_init(m, apply_to=linear_layers,
                         nonlinearity='relu', bias_init=0.,
                         **init_fn_params):

    nonlinearity = 'linear' if nonlinearity == 'identity' else nonlinearity
    if isinstance(m, apply_to):
        init.kaiming_uniform_(
            m.weight, nonlinearity=nonlinearity,
            **init_fn_params)
        if m.bias is not None:
            init.constant_(m.bias, bias_init)


def kaiming_normal_init(m, apply_to=linear_layers,
                        nonlinearity='relu', bias_init=0.,
                        **init_fn_params):

    nonlinearity = 'linear' if nonlinearity == 'identity' else nonlinearity
    if isinstance(m, apply_to):
        init.kaiming_normal_(
            m.weight, nonlinearity=nonlinearity,
            **init_fn_params)
        if m.bias is not None:
            init.constant_(m.bias, bias_init)


def normal_init(m, mean, std, bias_init=0., **init_fn_params):
    if isinstance(m, linear_layers):
        init.normal_(m.weight, mean, std)
        if m.bias is not None:
            init.constant_(m.bias, bias_init)


def get_init_linears_from_args(args, nonlinearity):
    name = args.get('name', 'xavier_uniform')
    bias_init = args.get('bias', 0.)
    init_fn_params = args.get('params', {})
    print("name init", name)
    print("bias init", bias_init)
    print("params init", **init_fn_params)

    return get_init_linears_fn(
        name, nonlinearity=nonlinearity,
        bias_init=bias_init,
        **init_fn_params)


def get_init_batch_norm_from_args(args):
    gamma_init = args.get('gamma', 1.)
    bias_init = args.get('bias', 0.)

    print("gamma init", gamma_init)
    print("bias init", bias_init)
    return partial(norm_layers_init,
                   gamma_init=gamma_init,
                   bias_init=bias_init)


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
        'linear': nn.Identity,
    })

    from packaging import version
    if version.parse(torch.__version__) >= version.parse("1.5"):
        activations.update({
            'hardsigmoid': activation.Hardsigmoid,
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


class CropImage(nn.Module):
    """Crops image to given size.
    Args:
        size
    """

    def __init__(self, size):
        super().__init__()
        self.size = size

    def forward(self, x):
        return crop_img_tensor(x, self.size)


def pad_img_tensor(x: torch.Tensor, size: Sequence[int]) -> torch.Tensor:
    """Pads a tensor.
    Pads a tensor of shape (batch, channels, h, w) to new height and width
    given by a tuple.
    Args:
        x (torch.Tensor): Input image
        size (list or tuple): Desired size (height, width)
    Returns:
        The padded tensor
    """

    return _pad_crop_img(x, size, 'pad')


def crop_img_tensor(x: torch.Tensor, size: Sequence[int]) -> torch.Tensor:
    """Crops a tensor.
    Crops a tensor of shape (batch, channels, h, w) to new height and width
    given by a tuple.
    Args:
        x (torch.Tensor): Input image
        size (list or tuple): Desired size (height, width)
    Returns:
        The cropped tensor
    """
    return _pad_crop_img(x, size, 'crop')


def _pad_crop_img(x: torch.Tensor, size: Sequence[int], mode: str) -> torch.Tensor:
    """ Pads or crops a tensor.
    Pads or crops a tensor of shape (batch, channels, h, w) to new height
    and width given by a tuple.
    Args:
        x (torch.Tensor): Input image
        size (list or tuple): Desired size (height, width)
        mode (str): Mode, either 'pad' or 'crop'
    Returns:
        The padded or cropped tensor
    """

    assert x.dim() == 4 and len(size) == 2
    size = tuple(size)
    x_size = x.size()[2:4]
    if mode == 'pad':
        cond = x_size[0] > size[0] or x_size[1] > size[1]
    elif mode == 'crop':
        cond = x_size[0] < size[0] or x_size[1] < size[1]
    else:
        raise ValueError("invalid mode '{}'".format(mode))
    if cond:
        raise ValueError('trying to {} from size {} to size {}'.format(
            mode, x_size, size))
    dr, dc = (abs(x_size[0] - size[0]), abs(x_size[1] - size[1]))
    dr1, dr2 = dr // 2, dr - (dr // 2)
    dc1, dc2 = dc // 2, dc - (dc // 2)
    if mode == 'pad':
        return nn.functional.pad(x, [dc1, dc2, dr1, dr2, 0, 0, 0, 0])
    elif mode == 'crop':
        return x[:, :, dr1:x_size[0] - dr2, dc1:x_size[1] - dc2]


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
