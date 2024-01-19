__version__ = "1.2.0"

from loguru import logger as console_logger
from typing import Union

from ._archive import archive_logs
from ._logger import TBLogger, WBLogger
from ._sync import sync

LoggerType = Union[TBLogger, WBLogger]

__all__ = ["LoggerType", "TBLogger", "WBLogger", "console_logger", "sync", "archive_logs"]
