from typing import Union

from loguru import logger as console_logger

from ..logger._archive import archive_logs
from ..logger._logger import TBLogger
from ..logger._plot import average_smooth, tb2dict, window_smooth
from ..logger._sync import download_logs, upload_logs

__all__ = [
    console_logger,
    archive_logs,
    TBLogger,
    upload_logs,
    download_logs,
    tb2dict,
    average_smooth,
    window_smooth,
]
