__version__ = "1.2.0"

from loguru import logger as console_logger

from ._archive import archive_logs
from ._logger import TBLogger, WBLogger
from ._sync import sync
from ._torch import load_torch_model, save_torch_model

__all__ = [
    "TBLogger",
    "WBLogger",
    "console_logger",
    "sync",
    "archive_logs",
    "save_torch_model",
    "load_torch_model",
]
