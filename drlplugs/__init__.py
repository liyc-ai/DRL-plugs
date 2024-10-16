import importlib.metadata

from drlplugs import drls, logger, net, ospy

__version__ = importlib.metadata.version("drlplugs")

__all__ = ["logger", "drls", "net", "ospy"]
