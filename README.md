# mllogger

An integrated logger for machine learning experiments.

## Installation

```bash
git clone https://github.com/BepfCp/mllogger
cd mllogger
pip install -e .
```

## Examples

We mainly support two types of  loggers:

+ TBLogger: combination of [[tensorboardX](https://github.com/lanpa/tensorboardX)];
+ WBLogger: [[wandb](https://github.com/wandb/wandb)].

For convience, we bind TBLogger (and WBLogger) with [[loguru](https://github.com/Delgan/loguru)] such that we don't need to configure the I/O logger. Please see [example](./example) for some illustrallations.
