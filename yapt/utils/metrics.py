import torch
import numpy as np
from collections import OrderedDict
from sklearn import metrics
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_fscore_support


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class AverageMeterDict:
    def __init__(self, names, fmt=':f'):
        assert isinstance(names, (list, tuple)), \
            "names should be a list, instead is a {}".format(type(names))

        self.meters = OrderedDict()
        for name in names:
            self.meters[name] = AverageMeter(name, fmt)
            self.name = self.meters[name]

    def reset(self):
        for key, val in self.meters.items():
            self.meters[key].reset()

    def update(self, vals):
        assert isinstance(vals, dict), \
            "vals should be a dict, instead is a {}".format(type(vals))
        for key, val in vals.items():
            self.meters[key].update(val)

    def summary(self):
        _val, _avg = OrderedDict(), OrderedDict()
        for key, meter in self.meters.items():
            _val[key] = meter.val
            _avg[key] = meter.avg
        return _val, _avg


class ScikitMetric(object):
    """Computes and stores the running confusion matrix"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def compute_metric(self):
        raise NotImplementedError("define the metric to be computed")

    def reset(self):
        self.count = 0
        self.preds = []
        self.probs = []
        self.labels = []

    def update(self, batch_labels, batch_probs):
        assert len(batch_probs) == len(batch_labels), \
            "Batch size should be the same for probs and labels"

        if isinstance(batch_probs, torch.Tensor):
            batch_probs = batch_probs.clone().cpu().data.numpy()

        if isinstance(batch_labels, torch.Tensor):
            batch_labels = batch_labels.clone().cpu().data.numpy()

        self.preds.append(batch_probs.argmax(1))
        self.probs.append(batch_probs)
        self.labels.append(batch_labels)
        self.count += len(batch_probs)
        self.compute_metric()

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ConfusionMatrix(ScikitMetric):
    def __init__(self, args, num_classes, **kwargs):
        super().__init__(args, **kwargs)
        self.num_classes = num_classes

    def reset(self):
        self.count = 0
        self.preds = []
        self.probs = []
        self.labels = []

        self.conf_matrix = []
        self.tn, self.fp, self.fn, self.tp = 0, 0, 0, 0
        self.precision, self.recall, self.fscore = 0, 0, 0
        self.accuracy = 0

    @property
    def val(self):
        return {
            'acc': self.accuracy,
            'prec': self.precision,
            'recall': self.recall,
            'fscore': self.fscore
        }

    @property
    def avg(self):
        return self.val

    def compute_metric(self):
        labels = np.concatenate(self.labels, axis=0)
        preds = np.concatenate(self.preds, axis=0)

        self.conf_matrix = confusion_matrix(
            labels, preds, labels=np.arange(self.num_classes))

        if self.num_classes == 2:
            self.tn, self.fp, self.fn, self.tp = self.conf_matrix.ravel()
            self.accuracy = (
                (self.tp * 1.0 + self.tn * 1.0) /
                (self.tp * 1.0 + self.fp * 1.0 + self.fn * 1.0 + self.tn * 1.0))

            (self.precision, self.recall,
             self.fscore, _) = precision_recall_fscore_support(
                labels, preds, average='binary', pos_label=1)


class ROC_AUC(ScikitMetric):
    def __init__(self, args, **kwargs):
        super().__init__(args, **kwargs)

    def reset(self):
        self.count = 0
        self.preds = []
        self.probs = []
        self.labels = []
        self.fpr, self.tpr, self.roc_auc = 0, 0, 0

    @property
    def val(self):
        return {'roc_auc': self.roc_auc}

    @property
    def avg(self):
        return self.val

    def compute_metric(self):
        labels = np.concatenate(self.labels, axis=0)
        probs = np.concatenate(self.probs, axis=0)

        self.fpr, self.tpr, _ = roc_curve(labels, probs[:, 1])
        self.roc_auc = auc(self.fpr, self.tpr)
