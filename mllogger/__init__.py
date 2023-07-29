__version__ = "1.0.1"

from loguru import logger as console_logger

from .logger import TBLogger, WBLogger

__all__ = ["TBLogger", "WBLogger", "console_logger"]
