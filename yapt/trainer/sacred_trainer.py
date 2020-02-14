import os
import sys
import functools
import inspect

from sacred import Experiment
from sacred.observers import MongoObserver
from sacred.utils import apply_backspaces_and_linefeeds
from yapt import BaseTrainer


def main_ifsacred(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        self = args[0]
        # Check at run-time if sacred has to be used
        if self.use_sacred:
            # If this is the case, wrap the function
            @self.sacred_exp.main
            def decor_func():
                return func(*args, **kwargs)

            # and run it through sacred run()
            self.sacred_exp.run()
        else:
            # Otherwise just run the function
            return func(*args, **kwargs)

    return wrapper


class SacredTrainer(BaseTrainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        args = self.args

        self.use_sacred = args.sacred.use_sacred
        if self.use_sacred:
            self.sacred_exp = Experiment(args.exp_name)
            self.sacred_exp.captured_out_filter = apply_backspaces_and_linefeeds
            self.sacred_exp.add_config(vars(args))
            for source in self.get_sources():
                self.sacred_exp.add_source_file(source)

            if not args.sacred.mongodb_disable:
                url = "{0.mongodb_url}:{0.mongodb_port}".format(args)
                if (args.sacred.mongodb_name is not None and
                        args.sacred.mongodb_name != ''):
                    db_name = args.sacred.mongodb_name
                else:
                    db_name = args.sacred.mongodb_prefix + ''.join(filter(
                        str.isalnum, args.dataset_name.lower()))

                print('Connect to MongoDB@{}:{}'.format(url, db_name))
                self.sacred_exp.observers.append(MongoObserver.create(url=url, db_name=db_name))

    def log_sacred_scalar(self, name, val, step):
        if self.use_sacred and self.sacred_exp.current_run:
            self.sacred_exp.current_run.log_scalar(name, val, step)

    def get_sources(self):
        sources = []
        # The network file
        sources.append(inspect.getfile(self.model.__class__))
        # the main script
        sources.append(sys.argv[0])
        # and any user custom submodule
        for module in self.model.children():
            module_path = inspect.getfile(module.__class__)
            if 'site-packages' not in module_path:
                sources.append(module_path)
        return sources

    @main_ifsacred
    def fit(self):
        super().fit()

    def json_results(self, savedir, testscore):
        super().json_results(savedir, testscore)
        json_path = os.path.join(savedir, "results.json")
        if self.use_sacred and self.sacred_exp.current_run:
            self.sacred_exp.current_run.current_run.add_artifact(json_path)
