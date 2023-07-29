from mllogger import TBLogger

args = {"description": "TEST", "lr": 3e-4}

record_param = [
    "description",
    "lr",
]  #  Used to name the log dir

logger = TBLogger(
    work_dir= "./",
    args=args, 
    root_log_dir="logs", 
    record_param=record_param,
    backup_code= True,
    code_files_list=["mllogger", "setup.py"]
)


"""
The output is like below, where `ckpt/` (gotten by `logger.ckpt_dir`) are used to save models, `result/` (gotten by logger.result_dir) are used to save outputs like images, `code/` (gotten by logger.code_bk_dir) are used to backup the code. If your want to manually add your own files and directionarie, you can access the current log dir by `logger.exp_dir`.

"""

# Tensorboard.
# For more apis, please see https://github.com/lanpa/tensorboardX
logger.tb.add_scalar(tag="train/return", scalar_value=10, global_step=0)

logger.add_dict({"loss": 0.5, "accuracy": 0.8}, t=0)

# loguru.
# For more apis, please see https://github.com/Delgan/loguru
logger.console.info("Hello, world!")
