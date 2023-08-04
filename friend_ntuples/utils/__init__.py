import os
import shutil


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


def iterate_files(source_base_dir, target_base_dir, files, processed_files_dir=".processed", restart=False):
    # Remove the source_base_dir prefix
    files = [file[len(source_base_dir):] if os.path.isabs(file) else file for file in files]

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

    # Read the list of already processed files
    processed_files_dir = os.path.join(target_base_dir, processed_files_dir)
    if not os.path.exists(processed_files_dir):
        os.makedirs(processed_files_dir)

    files = [file for file in files if not os.path.exists(os.path.join(processed_files_dir, file.replace("/", "_")))]

    # Loop over all the files
    for i_file, file in enumerate(files):
        source_path = os.path.join(source_base_dir, file)
        target_path = os.path.join(target_base_dir, file)
        processed_name = file.replace("/", "_")
        processed_path = os.path.join(processed_files_dir, processed_name)

        if os.path.exists(processed_path):
            print(f"File [{i_file + 1}/{len(files)}] {source_path} has already been processed, skipping...")
            continue

        print(f"File [{i_file + 1}/{len(files)}] {source_path} -> {target_path}")

        # Create the target directory if it doesn't exist
        target_dir = os.path.dirname(target_path)
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)

        yield i_file, source_path, target_path, processed_path
