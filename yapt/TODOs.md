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
- [x] should we use Tune logger? is it feasible? At least save in the same folder!
    - Yes, but the metrics you want to log should be returned in _train as result dictionary.
- [] add time statistics for each method as in ray sgd pytorch runner
- [] choose to plot gradients and norms
- [] integrate https://discuss.pytorch.org/t/check-gradient-flow-in-network/15063/6

## Parser
- [] make sure that type does not change in args and also handle missing params
- [] create .secret and handle sacred mongodb (username is default from userid) https://gitlab.com/airlab-404/server-guidelines/-/wikis/home

## Model Zoo
- [] create_episodes for meta datasets. where should it be called in yapt?
- [x] modify vae models for schedulers and optimizers

## DVAE
- [] run exploration parameters on cmnist
- [] check if I can run experiment with  mutual information (MINE)
-
-
## Tune
- [] check possible tune args from cli, otherwise it breaks
-(pid=175) --config-list: null
(pid=175) --node-ip-address: 172.17.0.18
(pid=175) --node-manager-port: 42041
(pid=175) --object-store-name: /tmp/ray/session_2020-02-17_23-18-57_895840_1/sockets/plasma_store
(pid=175) --raylet-name: /tmp/ray/session_2020-02-17_23-18-57_895840_1/sockets/raylet
(pid=175) --redis-address: 172.17.0.18:40064
(pid=175) --redis-password: null
(pid=175) --temp-dir: /tmp/ray
(pid=175) --use-pickle: null
(pid=175) '5241590000000000': null

