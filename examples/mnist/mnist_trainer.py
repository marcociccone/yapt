from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from yapt import Trainer
from model import Classifier


class TrainerMNIST(Trainer):

    default_config = '/exp/examples/mnist/mnist.yml'  # docker
    # default_config = 'examples/mnist/mnist.yml'        # colab

    def set_data_loaders(self):
        args = self.args
        kwargs = {'num_workers': args.data.num_workers,
                  'pin_memory': args.data.pin_memory} if args.cuda else {}

        data_loaders = dict()
        data_loaders['train'] = DataLoader(
            datasets.MNIST(args.data.datadir, train=True, download=True,
                           transform=transforms.ToTensor()),
            batch_size=args.batch_size, shuffle=True, **kwargs)

        data_loaders['val'] = data_loaders['test'] = DataLoader(
            datasets.MNIST(args.data.datadir, train=False,
                           transform=transforms.ToTensor()),
            batch_size=args.batch_size, shuffle=False, **kwargs)

        # TODO: should it be returned or set?
        # is the name correct, is a setter?
        return data_loaders


if __name__ == "__main__":

    trainer = TrainerMNIST(model_class=Classifier)
    trainer.fit()
