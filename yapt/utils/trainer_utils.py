import torch
import torch.nn as nn
import torch.nn.init as init
from torch import optim
from collections import OrderedDict


def xavier_uniform_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.xavier_uniform_(m.weight)
        if m.bias is not None:
            init.zeros_(m.bias)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        init.ones_(m.weight)
        if m.bias is not None:
            init.zeros_(m.bias)


def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal_(m.weight)
        if m.bias is not None:
            init.zeros_(m.bias)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        init.ones_(m.weight)
        if m.bias is not None:
            init.zeros_(m.bias)


def normal_init(m, mean, std):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.normal_(m.weight, mean, std)
        if m.bias is not None:
            init.zeros_(m.bias)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        init.ones_(m.weight)
        if m.bias is not None:
            init.zeros_(m.bias)


def alternate_datasets(labelled, unlabelled):
    num_steps = len(labelled) + len(unlabelled)
    every = int(len(labelled) / len(unlabelled) + 1)
    _unsup = 0

    for idx in range(num_steps):
        if (idx % every == 1 and _unsup < len(unlabelled)):
            _unsup += 1
            yield (next(unlabelled), False)
        else:
            yield (next(labelled), True)


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


def detach_tensor(tensor_list):
    """
    Helper function to recursively detach a tensor or a list of tensors
    This should be used when storing elements that could accumulate gradients.
    """
    if isinstance(tensor_list, (list, tuple)):
        new_tensors = []
        for tensor in tensor_list:
            if isinstance(tensor, torch.Tensor):
                new_tensors.append(tensor.detach())
            else:
                new_tensors.append(detach_tensor(tensor))

        return new_tensors

    elif isinstance(tensor_list, torch.Tensor):
        return tensor_list.detach()

    else:
        return tensor_list


def detach_dict(dict_tensor):
    """
    Helper function to recursively detach a dictionary of tensors
    This should be used when storing elements that could accumulate gradients.
    """
    for key, val in dict_tensor.items():
        if isinstance(val, dict):
            dict_tensor[key] = detach_dict(val)
        else:
            dict_tensor[key] = detach_tensor(val)

    return dict_tensor


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
