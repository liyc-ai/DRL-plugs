import os
from os.path import join

from rlplugs.ospy.file import copys


def archive_logs(src_log_dir: str, exp_name: str, tgt_dir: str = "archived"):
    """Archive logs named [exp_name] in [src_dir] to tgt_dir"""
    os.makedirs(tgt_dir, exist_ok=True)
    src_dir = join(src_log_dir, exp_name)
    assert os.path.exists(src_dir)
    copys(src_dir, join(tgt_dir, exp_name))
