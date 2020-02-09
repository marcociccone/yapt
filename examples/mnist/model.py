import torch
import torch.nn.functional as F
from torch import nn
from torch import optim

from collections import OrderedDict
from .yapf.base import BaseModel


class Classifier(BaseModel):

    def build_model(self):
        """
        Layout model
        :return:
        """
        args = self.args
        net_params = args.net_params

        self.c_d1 = nn.Linear(in_features=net_params.in_features,
                              out_features=net_params.hidden_dim)
        self.c_d1_bn = nn.BatchNorm1d(net_params.hidden_dim)
        self.c_d1_drop = nn.Dropout(net_params.drop_prob)

        self.c_d2 = nn.Linear(in_features=net_params.hidden_dim,
                              out_features=net_params.out_features)
    def forward(self, x):

        x = self.c_d1(x)
        x = torch.tanh(x)
        x = self.c_d1_bn(x)
        x = self.c_d1_drop(x)

        x = self.c_d2(x)
        logits = F.log_softmax(x, dim=1)

        return logits

    def configure_optimizers(self):
            """
            return whatever optimizers we want here
            :return: list of optimizers
            """
            args = self.args
            opt_params = args.optimizer.params
            optimizer = optim.Adam(self.parameters(), **opt_params.toDict())
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
            return [optimizer], [scheduler]

    def loss(self, labels, logits):
        nll = F.nll_loss(logits, labels)
        return nll

    def training_step(self, batch):

        # forward pass
        x, y = batch
        x = x.view(x.size(0), -1)

        y_hat = self.forward(x)

        # calculate loss
        loss_val = self.loss(y, y_hat)

        # in DP mode (default) make sure if result is scalar, there's another dim in the beginning
        if self.trainer.use_dp or self.trainer.use_ddp2:
            loss_val = loss_val.unsqueeze(0)

        tqdm_dict = {'train_loss': loss_val}
        output = OrderedDict({
            'loss': loss_val,
            'progress_bar': tqdm_dict,
            'log': tqdm_dict
        })

        # can also return just a scalar instead of a dict (return loss_val)
        return output

    def validation_step(self, batch):
        x, y = batch
        x = x.view(x.size(0), -1)
        y_hat = self.forward(x)

        loss_val = self.loss(y, y_hat)

        # acc
        labels_hat = torch.argmax(y_hat, dim=1)
        val_acc = torch.sum(y == labels_hat).item() / (len(y) * 1.0)
        val_acc = torch.tensor(val_acc)

        # in DP mode (default) make sure if result is scalar, there's another dim in the beginning
        if self.trainer.use_dp or self.trainer.use_ddp2:
            loss_val = loss_val.unsqueeze(0)
            val_acc = val_acc.unsqueeze(0)

        output = OrderedDict({
            'val_loss': loss_val,
            'val_acc': val_acc,
        })

        # can also return just a scalar instead of a dict (return loss_val)
        return output
