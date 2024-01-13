__version__ = "1.2.0"

from loguru import logger as console_logger

from ._archive import archive_logs
from ._logger import TBLogger, WBLogger
from ._sync import sync

__all__ = ["TBLogger", "WBLogger", "console_logger", "sync", "archive_logs"]
