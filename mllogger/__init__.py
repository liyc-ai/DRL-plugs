__version__ = "1.0.1"

from loguru import logger as console_logger

from ._logger import TBLogger, WBLogger
from ._sync import sync

__all__ = ["TBLogger", "WBLogger", "console_logger", "sync"]
