import os

from dotenv import load_dotenv

from mllogger import TBLogger, sync

args = {"description": "TEST", "lr": 3e-4}

record_param = [
    "description",
    "lr",
]  #  Used to name the log dir

logger = TBLogger(
    work_dir="./",
    args=args,
    root_log_dir="logs",
    record_param=record_param,
    backup_code=True,
    code_files_list=["mllogger", "setup.py"],
)


"""
The output is like below, where `ckpt/` (gotten by `logger.ckpt_dir`) are used to save models, `result/` (gotten by logger.result_dir) are used to save outputs like images, `code/` (gotten by logger.code_bk_dir) are used to backup the code. If your want to manually add your own files and directionarie, you can access the current log dir by `logger.exp_dir`.

.
├── 2023-07-29__13-00-15~description=TEST~lr=0.0003
│   ├── ckpt/
│   ├── code/
│   ├── console.log
│   ├── events.out.tfevents.1690606815.DESKTOP-HGHMVKR
│   ├── parameter.json
│   └── result/
└── 2023-07-29__13-01-24~description=TEST~lr=0.0003
    ├── ckpt/
    ├── code/
    ├── console.log
    ├── events.out.tfevents.1690606884.DESKTOP-HGHMVKR
    ├── parameter.json
    └── result/

"""

# Tensorboard.
# For more apis, please see https://github.com/lanpa/tensorboardX
logger.tb.add_scalar(tag="train/return", scalar_value=10, global_step=0)

logger.add_dict({"loss": 0.5, "accuracy": 0.8}, t=0)

# loguru.
# For more apis, please see https://github.com/Delgan/loguru
logger.console.info("Hello, world!")

logger.tb.close()

logger.archive(tgt_dir="archive")

# # sync code
# load_dotenv("./remote.env")
# """
# Content of remote.env:

# HOSTNAME = "xx.xx.xx.xx"
# PORT = 22
# USERNAME = "xxx"
# PASSWD = "xxxxxxx"
# REMOTE_WORK_DIR = "/path/to/logs"
# """
# sync(
#     hostname=os.environ["HOSTNAME"],
#     port=os.environ["PORT"],
#     username=os.environ["USERNAME"],
#     passwd=os.environ["PASSWD"],
#     remote_work_dir=os.environ["REMOTE_WORK_DIR"],
#     local_work_dir="./",
#     local_log_dir="logs",
# )
