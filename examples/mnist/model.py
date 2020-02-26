import torch
import torch.nn.functional as F
from torch import nn
from torch import optim

from collections import OrderedDict
from yapt import BaseModel
from yapt.utils.trainer_utils import get_optimizer
from yapt.utils.metrics import AverageMeter, ConfusionMatrix


class Classifier(BaseModel):

    def _build_model(self):
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

    def loss(self, labels, logits):
        return F.nll_loss(logits, labels)

    def _configure_optimizer(self):
        """
            This is the default implementation in BaseModel,
            I let it here just to show how this method should be implemented
        """
        args = self.args
        opt_params = self.args.optimizer.params
        weight_decay = self.args.optimizer.regularizers.weight_decay

        # -- Get optimizer from args
        opt_class = get_optimizer(args.optimizer.name)

        # -- Instantiate optimizer with specific parameters
        optimizer = opt_class(
            self.parameters(), weight_decay=weight_decay, **opt_params)
        return optimizer

    def _configure_scheduler_optimizer(self):
        # -- LR scheduler
        return optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=10)

    def _training_step(self, batch, epoch):

        # forward pass
        x, y = batch
        y_probs = self.forward(x)
        loss = self.loss(y, y_probs)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self._train_meters['loss'].update(loss)
        stats = {'loss': loss}

        running_tqdm, final_tqdm = OrderedDict(), OrderedDict()
        for key, meter in self._train_meters.items():
            running_tqdm[key] = meter.val
            final_tqdm[key] = meter.avg

        output = OrderedDict({
            'running_tqdm': running_tqdm,
            'final_tqdm': final_tqdm,
            'stats': stats
        })

        # TODO: not yet, but it will?
        # can also return just a scalar instead of a dict (return loss_val)
        return output

    def _validation_step(self, batch, epoch):
        with torch.no_grad():
            x, y = batch
            x = x.view(x.size(0), -1)

            y_probs = self.forward(x)
            val_loss = self.loss(y, y_probs)

            y_preds = y_probs.argmax(1)
            val_acc = torch.mean((y == y_preds).float())

            self._val_meters['loss'].update(val_loss)
            self._val_meters['acc'].update(val_acc)
            self._val_meters['cm'].update(y, y_probs)

            # -- This will be logged in tensorboard
            stats = {
                'loss': self._val_meters['loss'].avg,
                'acc': self._val_meters['acc'].avg,
            }

            output = OrderedDict({
                'final_tqdm': stats,
                'stats': stats,
            })

            # TODO: not yet, but it will?
            # can also return just a scalar instead of a dict (return loss_val)
            return output

    def _reset_train_stats(self):
        self._train_meters['loss'] = AverageMeter('loss')

    def _reset_val_stats(self):
        self._val_meters['acc'] = AverageMeter('acc')
        self._val_meters['loss'] = AverageMeter('loss')
        self._val_meters['cm'] = ConfusionMatrix('cm', self.args.net_params.out_dim)

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
