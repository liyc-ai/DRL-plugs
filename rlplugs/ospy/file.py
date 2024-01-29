import os
import re
import shutil
from typing import List


def copys(src_path: str, tgt_path: str):
    if os.path.isfile(src_path):
        shutil.copy(src_path, tgt_path)
    elif os.path.isdir(src_path):
        shutil.copytree(src_path, tgt_path)
    else:
        raise TypeError("Unknown code file type!")


def filter_from_list(file_list: List[str], rule: str) -> List[str]:
    """Could be used to match files given the [rule]"""
    return list(filter(lambda x: re.match(rule, x) != None, file_list))
