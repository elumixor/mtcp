import os
import shutil

from .file_lock import FileLock


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


def get_all_files(base_dir: str):
    """Gets the list of all the .root files in the base_dir and its subdirectories recursively"""
    files = []
    remaining = [base_dir]
    while remaining:
        current_dir = remaining.pop()

        for file in os.listdir(current_dir):
            file = os.path.join(current_dir, file)
            if file.endswith(".root"):
                files.append(file)
            elif os.path.isdir(file):
                remaining.append(file)

    return files


def iterate_files(source_base_dir, target_base_dir, files, restart=False):
    # Remove the source_base_dir prefix
    # Also filter out NoNeed dirs
    files = [file[len(source_base_dir):] if os.path.isabs(file) else file for file in files]

    # Create directories if they don't exist
    directories = set([os.path.dirname(file) for file in files])
    for directory in directories:
        if "NoNeed" in directory:
            continue

        target_directory = os.path.join(target_base_dir, directory)
        if not os.path.exists(target_directory):
            os.makedirs(target_directory)

    # Check if the source directory exists
    if not os.path.exists(source_base_dir):
        raise ValueError(f"Source directory {source_base_dir} does not exist")

    # Remove the target directory if we are restarting
    if restart:
        if os.path.exists(target_base_dir):
            print(f"Removing target directory {target_base_dir}")
            shutil.rmtree(target_base_dir)

    # Create the target directory if it doesn't exist
    if not os.path.exists(target_base_dir):
        print(f"Creating target directory {target_base_dir}")
        os.makedirs(target_base_dir)

    # Loop over all the files
    for i_file, file in enumerate(files):
        if "NoNeed" in file:
            print(f"NoNeed: {file}")
            continue

        source_path = os.path.join(source_base_dir, file)
        target_path = os.path.join(target_base_dir, file)

        yield i_file, file, source_path, target_path
