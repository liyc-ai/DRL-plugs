# MLG
An integrated logger for machine learning experiments.


## Installation

```bash
git clone https://github.com/BepfCp/mllogger
cd mllogger
pip install -e .
```

## Quickstart

```python

from mlg import IntegratedLogger

record_param = {
    "description": "TEST"
}  #  Used to name the log dir

logger = IntegratedLogger(record_param, log_root="logs")

# Tensorboard. 
# For more apis, please see https://github.com/lanpa/tensorboardX
logger.add_scalar(
    tag = "train/return",
    scalar_value = 10,
    global_step = 0
)

# loguru. 
# For more apis, please see https://github.com/Delgan/loguru
logger.info("Hello, world!")
```