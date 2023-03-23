from setuptools import find_packages, setup


def get_version():
    """Gets the imitation_base version."""
    path = "mlg/__init__.py"
    with open(path) as file:
        lines = file.readlines()

    for line in lines:
        if line.startswith("__version__"):
            return line.strip().split()[-1].strip().strip('"')
    raise RuntimeError("bad version data in __init__.py")


def read_requirements():
    # To generate requirements:
    # pipreqs . --ignore [ignored dir] --force
    with open("./requirements.txt", "r", encoding="utf-8") as f:
        install_requires = f.read().splitlines()
    return install_requires


requires = read_requirements()

setup(
    name="mlg",
    version=get_version(),
    description="An out-of-box integrated logger.",
    author="Yi-Chen Li",
    author_email="ychenli.X@gmail.com",
    url="https://github.com/BepfCp/mllogger",
    packages=find_packages(include=["mlg.*"]),
    python_requires=">=3.7",
    install_requires=requires,
)
