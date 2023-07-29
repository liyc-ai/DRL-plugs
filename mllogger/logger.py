import json
import os
import pprint
import types
from datetime import datetime
from os.path import exists, join
from typing import Any, Dict, List

import loguru
import wandb
from dotenv import load_dotenv
from tensorboardX import SummaryWriter


def _parse_record_param(
    args: Dict[str, Any], record_param: List[str]
) -> Dict[str, Any]:
    if args is None or record_param is None:
        return None
    else:
        record_param_dict = dict()
        for param in record_param:
            params = param.split(".")
            value = args
            for p in params:
                value = value[p]
            record_param_dict[param] = value
        return record_param_dict


def _get_exp_name(record_param_dict: Dict[str, Any], prefix: str = None):
    if prefix is not None:
        exp_name = prefix
    else:
        exp_name = datetime.now().strftime("%Y-%m-%d__%H-%M-%S")
    for key, value in record_param_dict.items():
        exp_name = exp_name + f"~{key}={value}"
    return exp_name


class TBLogger:
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
        self.record_param_dict = _parse_record_param(args, record_param)

        ## Do not change the following orders.
        self.exp_dir = join(self.root_log_dir, _get_exp_name(self.record_param_dict))
        self._create_artifact_dir()
        self._save_args()

        # init tb
        self.tb = SummaryWriter(log_dir=self.exp_dir, **kwargs)

        # init loguru
        self.console = loguru.logger
        self.console_log_file = join(self.exp_dir, "console.log")
        self.console.add(self.console_log_file, format="{time} -- {level} -- {message}")

    def _create_artifact_dir(self):
        self.ckpt_dir = join(self.exp_dir, "ckpt")
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
            self.tb.add_scalar(key, value, t)


class WBLogger:
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
        self.record_param_dict = _parse_record_param(args, record_param)
        self.wb = wandb.init(
            config=args,
            name=_get_exp_name(self.record_param_dict),
            project=project,
            entity=entity,
            **kwargs,
        )  # wandb.sdk.wandb_run.Run
        self.exp_dir = wandb.run.dir

        # init loguru logger
        self.console = loguru.logger
        self.console_log_file = join(self.exp_dir, "console.log")
        self.console.add(self.console_log_file, format="{time} -- {level} -- {message}")
