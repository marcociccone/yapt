import os
import logging as log
from logging import getLogger
from subprocess import check_output, CalledProcessError

try:
    cwd = os.path.join(__path__[0], os.path.pardir)
    __version__ = check_output('git rev-parse HEAD', cwd=cwd,
                               shell=True).strip().decode('ascii')
except CalledProcessError:
    __version__ = -1

__author__ = 'Marco Ciccone and Marco Cannici'
__author_email__ = 'marco.ciccone@polimi.it'
__license__ = 'Apache-2.0'
__copyright__ = 'Copyright (c) 2018-2020, %s.' % __author__
__homepage__ = 'https://gitlab.com/mciccone/yapt'
__docs__ = "YAPT: Yet Another PyTorch Trainer."

log.basicConfig(level=log.INFO)
_logger = getLogger("YAPT")

from .trainer.base import BaseTrainer
from .trainer.trainer import Trainer
from .trainer.tune_trainer import TuneWrapper, EarlyStoppingRule
from .core.model import BaseModel

__all__ = [
    'BaseTrainer',
    'Trainer',
    'BaseModel',
    'TuneWrapper',
    'EarlyStoppingRule'
]
