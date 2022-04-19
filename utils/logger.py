
import sys
import logging
from pathlib import Path
from typing import Union


def init_logger(name: str, path: Union[str, Path]):
    logger = logging.Logger(name)
    logger.addHandler(logging.FileHandler(str(path)))
    logger.addHandler(logging.StreamHandler(sys.stdout))
    return logger
