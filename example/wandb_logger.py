from mllogger import WBLogger

args = {"description": "TEST", "lr": 3e-4}
record_param = [
    "description",
    "lr",
]  #  Used to name the log dir

logger = WBLogger(
    args=args,
    record_param=record_param,
    project="mllogger",
    setting_file_path="./wandb.env",
)

logger.wb.log({"accuracy": 0.8, "return": "100."})

logger.wb.log_code(".")

logger.console.info("Hello, world!")
