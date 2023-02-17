from mlg import IntegratedLogger

record_param = {
    "description": "TEST"
}

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