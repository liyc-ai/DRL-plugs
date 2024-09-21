from drlplugs.logger import WBLogger

args = {"description": "TEST", "lr": 3e-4}
record_param = [
    "description",
    "lr",
]  #  Used to name the log dir

logger = WBLogger(
    args=args, record_param=record_param, project="mllogger", mode="offline"
)

# # or you can instantiate the wandb logger via a `.env` file, which may contain the following environment variables:

# """
# WANDB_PROJECT = "mllogger"
# WANDB_MODE = "online"
# WANDB_API_KEY = "xxx"
# WANDB_ENTITY = "xxx"
# """
# logger = WBLogger(
#     args=args, record_param=record_param, setting_file_path="./wandb.env"
# )

logger.wb.log({"accuracy": 0.8, "return": "100."})

logger.wb.log_code(".")

logger.console.info("Hello, world!")
