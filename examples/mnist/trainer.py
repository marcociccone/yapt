from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from yapt import Trainer
from yapt.configparser import parse_configuration
from .model import Classifier

class TrainerMNIST(Trainer):

    def set_data_loaders(self):
        args = self.args
        kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

        data_loaders = dict()
        data_loaders['train'] = DataLoader(
            datasets.MNIST(args.datadir, train=True, download=True,
                           transform=transforms.ToTensor()),
            batch_size=args.batch_size, shuffle=True, **kwargs)

        data_loaders['val'] = data_loaders['test'] = DataLoader(
            datasets.MNIST(args.datadir, train=False,
                           transform=transforms.ToTensor()),
            batch_size=args.batch_size, shuffle=False, **kwargs)

if __name__ == "__main__":

    default_config = "mnist.yml"
    args = parse_configuration(default_config, dump_config=False)

    trainer = TrainerMNIST(args=args, model_class=Classifier)
    trainer.fit()
