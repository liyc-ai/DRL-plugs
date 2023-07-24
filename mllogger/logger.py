import json
import os
import pprint
from datetime import datetime
from os.path import exists, join
from typing import Any, Dict, List

import wandb
from dotenv import load_dotenv
from loguru import logger
from tensorboardX import SummaryWriter
from wandb.sdk.wandb_run import Run


class TBLogger(SummaryWriter):
    """Tensorboard Logger"""

    def __init__(
        self,
        args: Dict[str, Any] = {},
        root_log_dir: str = "runs",
        record_param: List[str] = [],
        **kwargs,
    ):
        """
        Args:
            args: Hyper-parameters and configs
            root_log_dir: The root directory for all the logs
            record_param: Parameters used to name the log dir
        """
        self.args = args
        self.record_param = record_param
        self.root_log_dir = root_log_dir
        self.record_param_dict = self._parse_record_param(record_param)

        ## Do not change the following orders.
        self._create_exp_dir()
        self._create_ckpt_result_dir()
        self.save_args()

        super().__init__(log_dir=self.exp_dir, **kwargs)
        self.console_logger = logger
        self.console_log_file = join(self.exp_dir, "console_log.log")
        self.console_logger.add(
            self.console_log_file, format="{time} -- {level} -- {message}"
        )

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

    def save_args(self):
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
        args: Dict[str, Any] = {},
        record_param: List[str] = [],
        project: str = None,
        entity: str = None,
        setting_file_path: str = None,
        **kwargs,
    ):
        """
        Args:
            args: Hyper-parameters and configs
            record_param: Parameters used to name the log dir
            project: Name of the project
            entity: Username or team name
            setting_file_path: The `.env` file, for environment variables, see https://docs.wandb.ai/guides/track/environment-variables for more details.
        """
        if setting_file_path is not None and exists(setting_file_path):
            load_dotenv(setting_file_path)
        if kwargs.get("dir") and not exists(kwargs.get("dir")):
            os.makedirs(kwargs["dir"])
        # init wandb Run, https://docs.wandb.ai/ref/python/init
        wandb.init(
            config=args,
            name=self._parse_record_param(args, record_param),
            project=project,
            entity=entity,
            **kwargs,
        )
        self.exp_dir = wandb.run.dir

        # init loguru logger
        self.console_logger = logger
        self.console_log_file = join(self.exp_dir, "console_log.log")
        self.console_logger.add(
            self.console_log_file, format="{time} -- {level} -- {message}"
        )

    def _parse_record_param(self, args: Dict[str, Any], record_param: List[str]) -> str:
        if args is None or record_param is None:
            return None
        else:
            name = datetime.now().strftime("%Y-%m-%d__%H-%M-%S")
            for param in record_param:
                params = param.split(".")
                value = args
                for p in params:
                    value = value[p]
                name = name + f"~{param}={value}"
            return name
