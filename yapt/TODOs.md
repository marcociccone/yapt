# TODOs

## General
- [] discuss how to handle nograd in validation
- [] use collect dataset for both training and validation
- [] call on_epoch_end to decide what to do with outputs (collate)
- [] call on_validation_end to decide what to do with outputs (collate)
- [] rename stats in tb_scalars, tb_images and write doc about how to handle it.
- [] add tqdm_running dict and tqdm_final
- [] check schedulers.step best practices
- [x] add flag to disable tqdm

## Distributed

- https://github.com/ray-project/ray/blob/master/python/ray/experimental/sgd/pytorch/pytorch_runner.py
- https://github.com/ray-project/ray/blob/master/python/ray/experimental/sgd/pytorch/distributed_pytorch_runner.py
- https://github.com/ray-project/ray/blob/master/python/ray/experimental/sgd/pytorch/examples/tune_example.py
- https://github.com/ray-project/ray/blob/master/python/ray/experimental/sgd/pytorch/examples/train_example.py

## Logger
- [] should we use Tune logger? is it feasible? At least save in the same folder!
- [] add time statistics for each method as in ray sgd pytorch runner

## Parser
- [] make sure that type does not change in args and also handle missing params
- [] create .secret and handle sacred mongodb (username is default from userid) https://gitlab.com/airlab-404/server-guidelines/-/wikis/home

## Model Zoo
- [] create_episodes for meta datasets. where should it be called in yapt?
- [x] modify vae models for schedulers and optimizers

## DVAE
- [] run exploration parameters on cmnist
- [] check if I can run experiment with  mutual information (MINE)
