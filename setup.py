from setuptools import find_packages, setup


def get_version():
    """Gets the imitation_base version."""
    path = "mllogger/__init__.py"
    with open(path) as file:
        lines = file.readlines()

    for line in lines:
        if line.startswith("__version__"):
            return line.strip().split()[-1].strip().strip('"')
    raise RuntimeError("bad version data in __init__.py")


setup(
    name="mllogger",
    version=get_version(),
    description="An out-of-box integrated logger.",
    author="Yi-Chen Li",
    author_email="ychenli.X@gmail.com",
    url="https://github.com/BepfCp/mllogger",
    packages=find_packages(include=["mllogger*"]),
    python_requires=">=3.7",
    install_requires=[
        "loguru",
        "python-dotenv",
        "setuptools",
        "tensorboardX",
        "wandb"
    ],
)
