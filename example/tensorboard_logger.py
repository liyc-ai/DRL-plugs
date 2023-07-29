from mllogger import TBLogger

args = {"description": "TEST", "lr": 3e-4}

record_param = [
    "description",
    "lr",
]  #  Used to name the log dir

logger = TBLogger(args=args, root_log_dir="logs", record_param=record_param)


"""
The output is like below, where `checkoutpoint` (gotten by `logger.ckpt_dir`) are used to save models and `result` (gotten by logger.result_dir) are used to save outputs like images. If your want to manually add your own files and directionarie, you can access the current log dir by `logger.exp_dir`.

logs
├── 2023-03-28_22-10-07&description=TEST
│   ├── checkpoint
│   ├── events.out.tfevents.1680012607.DESKTOP-HGHMVKR
│   ├── console_log.log
│   ├── parameter.json
│   └── result
└── 2023-03-28_22-20-07&description=TEST&lr=0.0003
    ├── checkpoint
    ├── events.out.tfevents.1680013207.DESKTOP-HGHMVKR
    ├── console_log.log
    ├── parameter.json
    └── result
"""

# Tensorboard.
# For more apis, please see https://github.com/lanpa/tensorboardX
logger.add_scalar(tag="train/return", scalar_value=10, global_step=0)

logger.add_dict({"loss": 0.5, "accuracy": 0.8}, t=0)

# loguru.
# For more apis, please see https://github.com/Delgan/loguru
logger.console.info("Hello, world!")
