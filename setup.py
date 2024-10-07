from setuptools import find_packages, setup


def get_version():
    """Gets the rlplugs version."""
    path = "drlplugs/__init__.py"
    with open(path) as file:
        lines = file.readlines()

    for line in lines:
        if line.startswith("__version__"):
            return line.strip().split()[-1].strip().strip('"')
    raise RuntimeError("bad version data in __init__.py")


setup(
    name="drlplugs",
    version=get_version(),
    description="An out-of-box toolbox that integrates helpers for RL.",
    author="Yi-Chen Li",
    author_email="ychenli.X@gmail.com",
    url="https://github.com/liyc-ai/DRL-plugs",
    packages=find_packages(include=["drlplugs*"]),
    python_requires=">=3.7",
    install_requires=[
        "loguru",
        "python-dotenv",
        "setuptools",
        "tensorboardX",
        "wandb",
        "paramiko",
        "tqdm",
        "tensorboard",
        "pandas",
        "seaborn",
        "matplotlib",
        "numpy",
	    "numba"
    ],
)
