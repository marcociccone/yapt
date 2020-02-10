import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch import optim

from collections import OrderedDict
from yapt import BaseModel
from yapt.utils.trainer_utils import get_optimizer
from yapt.utils.metrics import AverageMeter, ConfusionMatrix


class Classifier(BaseModel):

    def build_model(self):
        args = self.args

        c, h, w = args.input_dims
        in_dim = c*h*w
        hidden_dim = args.net_params.hidden_dim
        out_dim = args.net_params.out_dim
        drop_prob = args.net_params.drop_prob

        self.MLP = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(drop_prob),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.MLP(x)
        logits = F.log_softmax(x, dim=1)

        return logits

    def configure_optimizers(self):
        args = self.args
        opt_params = self.args.optimizer.params.toDict()
        weight_decay = self.args.optimizer.regularizers.weight_decay

        # -- Get optimizer from args
        optimizer = get_optimizer(args.optimizer.name)

        # -- Instantiate optimizer with specific parameters
        self.optimizer = optimizer(
            self.parameters(), weight_decay=weight_decay, **opt_params)

        # -- LR scheduler
        self.optimizers_schedulers = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=10)

    def loss(self, labels, logits):
        return F.nll_loss(logits, labels)

    def training_step(self, batch):

        # forward pass
        x, y = batch
        y_probs = self.forward(x)
        loss = self.loss(y, y_probs)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # TODO: I don't like this has to be reimplemented, should be handled automagically
        self.meters['loss'].update(loss)
        self.running_batches += 1
        self.steps += 1

        stats = {'loss': loss}

        # TODO: string gives greater flexibility, but may it could be just a dict
        running_tqdm = 'loss : {:.4f}'.format(self.meters['loss'].val)
        final_tqdm = 'avg loss : {:.4f}'.format(self.meters['loss'].avg)

        output = OrderedDict({
            'running_tqdm': running_tqdm,
            'final_tqdm': final_tqdm,
            'stats': stats
        })

        # TODO: not yet, but it will?
        # can also return just a scalar instead of a dict (return loss_val)
        return output

    def validation_step(self, batch, epoch):
        x, y = batch
        x = x.view(x.size(0), -1)

        y_probs = self.forward(x)
        val_loss = self.loss(y, y_probs)

        y_preds = y_probs.argmax(1)
        val_acc = torch.mean((y == y_preds).float())

        self.running_val_batches += 1
        self.meters_val['val_loss'].update(val_loss)
        self.meters_val['val_acc'].update(val_acc)
        self.meters_val['val_cm'].update(y, y_probs)

        # -- This will be logged in tensorboard
        stats = {
            'loss': self.meters_val['val_loss'].avg,
            'acc': self.meters_val['val_acc'].avg,
        }

        final_tqdm = ''
        for key, val in stats.items():
            final_tqdm += "{}: {:.4f} - ".format(key, val)

        output = OrderedDict({
            'final_tqdm': final_tqdm,
            'stats': stats,
        })

        # TODO: not yet, but it will?
        # can also return just a scalar instead of a dict (return loss_val)
        return output

    def init_train_stats(self):
        self.reset_train_stats()

    def reset_train_stats(self):
        self.meters = dict()
        self.meters['loss'] = AverageMeter('loss')
        self.running_batches = 0

    def init_val_stats(self):
        self.reset_val_stats()

    def reset_val_stats(self):
        self.meters_val = dict()
        self.meters_val['val_acc'] = AverageMeter('val_acc')
        self.meters_val['val_loss'] = AverageMeter('val_loss')
        self.meters_val['val_cm'] = ConfusionMatrix('val_cm', self.args.net_params.out_dim)
        self.running_val_batches = 0

    def early_stopping(self, current_stats: dict, best_stats: dict) -> bool:
        # TODO: I don't like the way is done here, too many dictionaries
        # Maybe this could be the default one in the BaseModel object

        dataset = self.args.early_stopping.dataset
        metric = self.args.early_stopping.metric

        # -- Handle bad cases
        if dataset == '' or dataset is None or metric == '' or metric is None:
            return True, 9999
        if current_stats.get(dataset, None) is None:
            return True, 9999
        if current_stats[dataset]['stats'].get(metric, None) is None:
            return True, 9999

        current = current_stats[dataset]['stats'][metric]
        # -- first time
        if best_stats.get(dataset, None) is None:
            is_best = True
        else:
            best = best_stats[dataset]['stats'][metric]
            is_best = current > best

        return (is_best, current if is_best else best)


