import os
from os.path import join

import paramiko

# def _sync(sftp: paramiko.SFTPClient, local_path: str, remote_path: str):
#     try:
#         sftp.mkdir(remote_path)
#     except IOError:
#         pass  # The file exists

#     for item in os.listdir(local_path):
#         item_local_path = os.path.join(local_path, item)
#         item_remote_path = os.path.join(remote_path, item)

#         if os.path.isfile(item_local_path):
#             sftp.put(item_local_path, item_remote_path)
#         elif os.path.isdir(item_local_path):
#             _sync(sftp, item_local_path, item_remote_path)


# def sync(
#     hostname: str,
#     port: int,
#     username: str,
#     passwd: str,
#     remote_work_dir: str,
#     local_work_dir: str = "./",
#     local_folder_name: str = "logs",
# ):
#     """
#     Args:
#         hostname: IP address of the remote server
#         port: Port of the SSH
#         username: Your username on the remote server
#         passwd: Your corresponding password of the username

#     WARNING: KEEP YOUR PASSWD SECRET!!!
#     """
#     client = paramiko.SSHClient()
#     client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
#     client.connect(hostname, port, username, passwd)
#     sftp = client.open_sftp()

#     _sync(
#         sftp,
#         join(local_work_dir, local_folder_name),
#         join(remote_work_dir, local_folder_name),
#     )

#     print("Successfully sync code!")


def _sync(
    sftp: paramiko.SFTPClient,
    local_work_dir: str,
    local_log_dir: str,
    remote_work_dir: str,
):
    if local_log_dir not in sftp.listdir(remote_work_dir):
        remote_work_dir = join(remote_work_dir, local_log_dir)
        sftp.mkdir(remote_work_dir)

    local_work_dir = join(local_work_dir, local_log_dir)
    for item in os.listdir(local_work_dir):
        item_local_path = os.path.join(local_work_dir, item)
        item_remote_path = os.path.join(remote_work_dir, item)

        if os.path.isfile(item_local_path):
            sftp.put(item_local_path, item_remote_path)
            # print(f"Syncing {item_local_path} to {item_remote_path}...")
        elif os.path.isdir(item_local_path):
            _sync(sftp, local_work_dir, item, remote_work_dir)


def sync(
    hostname: str,
    port: int,
    username: str,
    passwd: str,
    remote_work_dir: str,
    local_work_dir: str = "./",
    local_log_dir: str = "logs",
):
    """
    Args:
        hostname: IP address of the remote server
        port: Port of the SSH
        username: Your username on the remote server
        passwd: Your corresponding password of the username

    WARNING: KEEP YOUR PASSWD SECRET!!!
    """
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(hostname, port, username, passwd)
    sftp = client.open_sftp()

    _sync(sftp, local_work_dir, local_log_dir, remote_work_dir)

    print("Successfully sync code!")
