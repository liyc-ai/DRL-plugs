# MLLOGGER
An integrated logger for machine learning experiments.


## Installation

```bash
git clone https://github.com/BepfCp/mllogger
cd mllogger
pip install -e .
```

## Quickstart

```python
from mllogger import IntegratedLogger

record_param = [
    "description",
    "lr",
]  #  Used to name the log dir

args = {
    "description": "TEST",
    "lr": 3e-4
}

logger = IntegratedLogger(record_param=record_param, log_root="logs", args=args)

# Tensorboard. 
# For more apis, please see https://github.com/lanpa/tensorboardX
logger.add_scalar(
    tag = "train/return",
    scalar_value = 10,
    global_step = 0
)

logger.add_dict(
    {
        "loss": 0.5,
        "accuracy": 0.8
    },
    t = 0
)

# loguru. 
# For more apis, please see https://github.com/Delgan/loguru
logger.info("Hello, world!")
```

The output is like below, where `checkoutpoint` (gotten by `logger.ckpt_dir`) are used to save models and `result` (gotten by logger.result_dir) are used to save outputs like images. If your want to manually add your own files and directionarie, you can access the current log dir by `logger.exp_dir`.

```bash
logs
├── 2023-03-28_22-10-07&description=TEST
│   ├── checkpoint
│   ├── events.out.tfevents.1680012607.DESKTOP-HGHMVKR
│   ├── log.log
│   ├── parameter.json
│   └── result
└── 2023-03-28_22-20-07&description=TEST&lr=0.0003
    ├── checkpoint
    ├── events.out.tfevents.1680013207.DESKTOP-HGHMVKR
    ├── log.log
    ├── parameter.json
    └── result
```