from mllogger import WBLogger

args = {"description": "TEST", "lr": 3e-4}

logger = WBLogger(config=args, project="mllogger", name="test_wandb_logger")

logger.log({"accuracy": 0.8, "return": "100."})

logger.console_logger.info("Hello, world!")
