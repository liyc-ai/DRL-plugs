from typing import Union

from loguru import logger as console_logger

from rlplugs.logger._archive import archive_logs
from rlplugs.logger._logger import TBLogger, WBLogger
from rlplugs.logger._plot import average_smooth, tb2dict, window_smooth
from rlplugs.logger._sync import download_logs, upload_logs

LoggerType = Union[TBLogger, WBLogger]

__all__ = [
    LoggerType,
    console_logger,
    archive_logs,
    TBLogger,
    WBLogger,
    upload_logs,
    download_logs,
    tb2dict,
    average_smooth,
    window_smooth,
]
