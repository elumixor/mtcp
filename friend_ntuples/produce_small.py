import autorootcwd  # Do not delete - adds the root of the project to the path

import os
import uproot
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing
from threading import Lock
import numpy as np

from argparse import ArgumentParser

from pipeliner.utils import read_yaml
from friend_ntuples.utils import iterate_files, str2bool, get_all_files, FileLock
from data_processing.processing import ConfigParser


parser = ArgumentParser()
parser.add_argument("--config", default="./friend_ntuples/config.yaml", help="Path to the config file")
parser.add_argument("--restart", action="store_true", help="Restart from the beginning")
parser.add_argument("--source-base-dir", help="Override the source base directory specified in the config file")
parser.add_argument("--target-base-dir", help="Override the target base directory specified in the config file")
parser.add_argument("--include-original", type=str2bool, help="Include the original branches in the output file")
parser.add_argument("--file", help="Process only the specified file")

args = parser.parse_args()

# Read the config file
if args.config:
    config = read_yaml(args.config)
else:
    config = {}

# Get the cut expression from the trex config
parser = ConfigParser(config.trex_config)
cut_expression = parser.cut_expr(config.region)


# Loop over all the files
source_base_dir = args.source_base_dir if args.source_base_dir else config.source_base_dir
target_base_dir = args.target_base_dir if args.target_base_dir else os.path.join(config.target_base_dir, "small")
files = [args.file] if args.file else config.files if "files" in config else get_all_files(source_base_dir)

global_lock_path = os.path.join(target_base_dir, ".all-lock")



def process_file(i_file, file_name, source_path, target_path):
    try:
        file_str = f"File [{i_file + 1}/{len(files)}]"

        processed_path = f"{target_path}.processed"
        lock_path = f"{target_path}.lock"

        # if os.path.exists(processed_path):
        #     print(f"File [{i_file + 1}/{len(files)}] {file_name} has already been processed, skipping...")
        #     return

        # with FileLock(global_lock_path):
        # Check if the lock file exists
        # if os.path.exists(lock_path):
        #     print(f"File [{i_file + 1}/{len(files)}] {file_name} is being processed by another process, skipping...")
        #     return

        # Create the lock file
        # with open(lock_path, "w") as f:
            # pass

        print(f"{file_str} {source_path} -> {target_path}")

        # Open the file
        with uproot.open(source_path) as source_file:
            # Read the list of trees
            trees = source_file.keys()

            # We need to join the individual cycles
            trees = list(set([tree.split(";")[0] for tree in trees]))
            result = {}

            # Loop over all the trees
            for i_tree, tree_name in enumerate(trees):
                tree = source_file[tree_name]
                n_events = tree.num_entries

                print(f"{file_str} {file_name}: [{i_tree + 1}/{len(trees)}] {tree_name} ({n_events} events)")

                tree_keys = tree.keys()

                # Now we should apply the cut and save the resulting tree
                if tree.num_entries == 0:
                    print(f"{file_str} {file_name}: [{i_tree + 1}/{len(trees)}] {tree_name} has no events")
                    result[tree_name] = {key: np.array([]) for key in features} if len(tree_keys) > 0 else {"dummy": np.array([])}
                else:
                    # Apply the cut and save a new file
                    arrays = tree.arrays(tree_keys, cut=cut_expression)

                    features = {}
                    for feature in tree_keys:
                        array = arrays[feature]
                        if len(array) == 0:
                            array = array.to_numpy()

                        features[feature] = array

                    result[tree_name] = features

        with uproot.recreate(target_path) as target_file:
            for tree_name, arrays in result.items():
                target_file[tree_name] = arrays

        # with FileLock(global_lock_path):
        # Add the file to the list of processed files (create an empty file)
        with open(processed_path, "w") as f:
            pass

        # Remove the lock file
        if os.path.exists(lock_path):
            os.remove(lock_path)

        print(f"{file_str} {file_name}: Done")
    except Exception as e:
        if os.path.exists(lock_path):
            os.remove(lock_path)

        print(f"{file_str}: ERROR: {e}")
        raise


if len(files) == 1:
    for i_file, file_name, source_path, target_path in iterate_files(source_base_dir, target_base_dir, files, restart=args.restart):
        process_file(i_file, file_name, source_path, target_path)
else:
    for i_file, file_name, source_path, target_path in iterate_files(source_base_dir, target_base_dir, files, restart=args.restart):
        process_file(i_file, file_name, source_path, target_path)

    # with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count() * 2) as executor:
    #     try:
    #         futures = []
    #         for i_file, file_name, source_path, target_path in iterate_files(source_base_dir, target_base_dir, files, restart=args.restart):
    #             future = executor.submit(process_file, i_file, file_name, source_path, target_path)
    #             futures.append(future)

    #         for future in as_completed(futures):
    #             try:
    #                 future.result()
    #             except Exception as e:
    #                 print(f"Error processing file: {e}")
    #     except KeyboardInterrupt:
    #         print("Interrupted")
    #         executor.shutdown(wait=False, cancel_futures=True)
    #         raise
