import shutil
from os.path import exists

from mllogger import TBLogger, WBLogger, console_logger

args = {"a": 1, "b": 2}
root_log_dir = "_logs_test_"
record_param = ["a"]
project = "test_wb_logger"
entity = "liyc-group"


def test_console_logger():
    console_logger.info("Hello, world!")


def test_tb_logger():

    logger = TBLogger(args=args, root_log_dir=root_log_dir, record_param=record_param)

    assert exists(logger.exp_dir)
    assert exists(logger.result_dir)
    assert exists(logger.ckpt_dir)
    assert exists(logger.console_log_file)

    logger.close()

    shutil.rmtree(root_log_dir)


def test_wb_logger():
    logger = WBLogger(
        args=args,
        record_param=record_param,
        project=project,
        entity=entity,
        setting_file_path="./wandb.env",
        dir=root_log_dir,
    )
    assert exists(logger.console_log_file)

    logger.finish()

    shutil.rmtree(root_log_dir)
