import atexit
import json
import os
import pprint
import sys
from datetime import datetime
from os.path import join
from typing import Any, Dict, List

import loguru
from loguru import _defaults
from loguru._logger import Core as _Core
from loguru._logger import Logger as _Logger
from tensorboardX import SummaryWriter
from wandb.sdk.wandb_run import Run

import wandb


class TBLogger(_Logger, SummaryWriter):
    def __init__(
        self,
        args: Dict = None,
        record_param: List[str] = None,
        root_log_dir: str = "logs",
        **kwargs,
    ):
        self.args = args
        self.record_param = record_param
        self.root_log_dir = root_log_dir
        self.record_param_dict = self._parse_record_param(record_param)

        ## Do not change the following orders.
        self._create_exp_dir()
        self._create_ckpt_result_dir()
        self._save_args()

        # init loguru, copied from loguru.__init__.py
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
            patchers=[],
            extra={},
        )

        if _defaults.LOGURU_AUTOINIT and sys.stderr:
            self.add(sys.stderr)
        atexit.register(self.remove)

        self.add(join(self.exp_dir, "log.log"), format="{time} -- {level} -- {message}")

        # init tensorboard writter
        SummaryWriter.__init__(self, log_dir=self.exp_dir, **kwargs)

    def _parse_record_param(self, record_param: List[str]) -> Dict[str, Any]:
        if self.args is None or record_param is None:
            return None
        else:
            record_param_dict = dict()
            for param in record_param:
                param = param.split(".")
                value = self.args
                for p in param:
                    value = value[p]
                record_param_dict["-".join(param)] = value
            return record_param_dict

    def _create_exp_dir(self):
        self.exp_dir = join(
            self.root_log_dir, datetime.now().strftime("%Y-%m-%d__%H-%M-%S")
        )
        if self.record_param_dict is not None:
            for key, value in self.record_param_dict.items():
                self.exp_dir = self.exp_dir + f"~{key}={value}"

    def _create_ckpt_result_dir(self):
        self.ckpt_dir = join(self.exp_dir, "checkpoint")
        os.makedirs(self.ckpt_dir)  # checkpoint, for model, data, etc.

        self.result_dir = join(self.exp_dir, "result")
        os.makedirs(self.result_dir)  # result, for some intermediate result

    def _save_args(self):
        if self.args is None:
            return
        else:
            pp = pprint.PrettyPrinter(indent=4)
            pp.pprint(self.args)
            with open(join(self.exp_dir, "parameter.json"), "w") as f:
                jd = json.dumps(self.args, indent=4)
                print(jd, file=f)

    # ================ Additional Helper Functions ================

    def add_dict(self, info: Dict[str, float], t: int):
        for key, value in info.items():
            self.add_scalar(key, value, t)


class WBLogger(Run):
    def __init__(
        self,
        config: Dict = {},
        project: str = None,
        entity: str = None,
        name: str = None,
        dir: str = None,
        **kwargs,
    ):
        """
        :param config: dict of hyper-paramters
        :param project: name of the project
        :param entity: username or team name
        :param name: name of this run
        :param dir: root dir of configs
        """
        # init wandb Run, https://docs.wandb.ai/ref/python/init
        wandb.init(
            dir=dir, config=config, project=project, entity=entity, name=name, **kwargs
        )
        self.exp_dir = wandb.run.dir

        # init loguru logger
        self.console_logger = loguru.logger
        self.console_logger.add(
            join(self.exp_dir, "log.log"), format="{time} -- {level} -- {message}"
        )
