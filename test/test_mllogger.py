import shutil

from mllogger import IntegratedLogger

LOGS = "logs_test"
import os


def test_logger():
    record_param = ["description"]  #  Used to name the log dir

    args = {"description": "TEST"}

    logger = IntegratedLogger(record_param=record_param, log_root=LOGS, args=args)

    assert os.path.exists(logger.ckpt_dir)
    assert os.path.exists(logger.result_dir)
    assert os.path.exists(os.path.join(logger.logdir, "parameter.json"))
    assert os.path.exists(os.path.join(logger.logdir, "log.log"))

    # Tensorboard.
    # For more apis, please see https://github.com/lanpa/tensorboardX
    logger.add_scalar(tag="train/return", scalar_value=10, global_step=0)

    logger.add_dict({"loss": 0.5, "accuracy": 0.8}, t=0)

    # loguru.
    # For more apis, please see https://github.com/Delgan/loguru
    logger.info("Hello, world!")

    shutil.rmtree(LOGS)
