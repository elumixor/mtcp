import autorootcwd  # Do not delete - adds the root of the project to the path

import os
import uproot

from argparse import ArgumentParser

from pipeliner.utils import read_yaml
from friend_ntuples.utils import iterate_files, str2bool, get_all_files
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
for i_file, source_path, target_path, processed_path in iterate_files(source_base_dir, target_base_dir, files, restart=args.restart):
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

            print(f"\tTree [{i_tree + 1}/{len(trees)}] {tree_name} ({n_events} events)")

            # Now we should apply the cut and save the resulting tree
            if tree.num_entries == 0:
                print(f"\t\tWARNING: {tree_name} has no events")

            # Apply the cut and save a new file
            arrays = tree.arrays(tree.keys(), cut=cut_expression)
            result[tree_name] = {feature: arrays[feature] for feature in tree.keys()}

    with uproot.recreate(target_path) as target_file:
        for tree_name, arrays in result.items():
            target_file[tree_name] = arrays

    # Add the file to the list of processed files (create an empty file)
    with open(processed_path, "w") as f:
        pass
