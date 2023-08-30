__version__ = "1.1.0"

from loguru import logger as console_logger

from ._archive import archive, load_model, save_model
from ._logger import TBLogger, WBLogger
from ._sync import sync

__all__ = [
    "TBLogger",
    "WBLogger",
    "console_logger",
    "sync",
    "archive",
    "save_model",
    "load_model",
]
