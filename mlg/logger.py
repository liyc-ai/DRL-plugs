import atexit as _atexit
import os
import sys as _sys
from datetime import datetime
from os.path import join
from typing import Dict

from loguru import _defaults
from loguru._logger import Core as _Core
from loguru._logger import Logger as _Logger
from tensorboardX import SummaryWriter


class IntegratedLogger(_Logger, SummaryWriter):
    def __init__(self, record_param: Dict[str, float] = None, log_root: str = "logs"):
        """
        :param record_param: Used for name the experiment results dir
        :param log_root: The root path for all logs
        """
        # loguru.logger, copied from loguru.__init__.py
        _Logger.__init__(
            self,
            core=_Core(),
            exception=None,
            depth=0,
            record=False,
            lazy=False,
            colors=False,
            raw=False,
            capture=True,
            patcher=None,
            extra={}
        )

        if _defaults.LOGURU_AUTOINIT and _sys.stderr:
            self.add(_sys.stderr)

        _atexit.register(self.remove)

        # Hyperparam
        self.log_root = log_root
        self.record_param = record_param

        # Do not change the following orders.
        self._create_print_logger()
        self._create_ckpt_result_dir()

        # Init SummaryWriter
        SummaryWriter.__init__(self, logdir=self.exp_dir)

    def _create_print_logger(self):
        self.exp_dir = join(self.log_root, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        if self.record_param is not None:
            for key, value in self.record_param.items():
                self.exp_dir = self.exp_dir + f"&{key}={value}"
        self.add(
            join(self.exp_dir, "log.log"), format="{time} -- {level} -- {message}"
        )

    def _create_ckpt_result_dir(self):
        self.ckpt_dir = join(self.exp_dir, "checkpoint")
        os.makedirs(self.ckpt_dir)  # checkpoint, for model, data, etc.

        self.resutl_dir = join(self.exp_dir, "result")
        os.makedirs(self.resutl_dir)  # result, for some intermediate result
