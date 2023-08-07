import autorootcwd  # Do not delete - adds the root of the project to the path

from array import array
import os
import sys
import numpy as np
import shutil
import importlib
import uproot
from tqdm import tqdm

from argparse import ArgumentParser
from friend_ntuples.utils import iterate_files, str2bool, get_all_files
from ml.utils import get_config
from pipeliner.utils import read_yaml

parser = ArgumentParser()
parser.add_argument("--config", default="~/mtcp/friend_ntuples/config-friend.yaml", help="Path to the config file")
parser.add_argument("--restart", action="store_true", help="Restart from the beginning")
parser.add_argument("--source-base-dir", help="Override the source base directory specified in the config file")
parser.add_argument("--target-base-dir", help="Override the target base directory specified in the config file")
parser.add_argument("--include-original", type=str2bool, help="Include the original branches in the output file")
parser.add_argument("--model-path", help="Path to the custom model file")
parser.add_argument("--branch-name", help="Name of the branch to add", default="probs_ttH")
parser.add_argument("--file", help="Process only the specified file")

args = parser.parse_args()

# Read the config file
if args.config:
    config = read_yaml(os.path.expanduser(args.config))
else:
    config = {}

added_branch_name = args.branch_name if args.branch_name else config["branch_name"]
model_path = args.model_path if args.model_path else config["model_path"]
batch_size = config.batch_size
include_original = args.include_original if args.include_original is not None else False

# Import ROOT if we need to include original file
if include_original:
    import ROOT

# Load the model
model_path = os.path.expanduser(model_path)
spec = importlib.util.spec_from_file_location("model", model_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
model = module.Model(config)
print("Model loaded successfully")

# Get the features used
features = model.features
print(f"Using {len(features)} features")

# Loop over all the files
source_base_dir = args.source_base_dir if args.source_base_dir else config.source_base_dir
target_base_dir = args.target_base_dir if args.target_base_dir else config.target_base_dir
files = [args.file] if args.file else config.files if "files" in config else get_all_files(source_base_dir)

run_name = get_config(config.model, check_cuda_device=False, silent=True).run_name

# Insert the run name into the target base dir after the friend_ntuples/output/friend
before, after = target_base_dir.split("friend_ntuples/output/friend/")
print(f"before: {before}")
print(f"after: {after}")
target_base_dir = os.path.join(before, "friend_ntuples/output/friend/", run_name, after)
print(f"target_base_dir: {target_base_dir}")

failed_path = os.path.join(os.path.expanduser(target_base_dir), "failed.txt")

# Recreate the failed file so that it's empty
if os.path.exists(failed_path):
    os.remove(failed_path)

failed = []


def process_file(i_file, file_name, source_path, target_path):
    file_str = f"File [{i_file + 1}/{len(files)}]"

    try:
        processed_path = f"{target_path}.processed"

        if os.path.exists(processed_path):
            print(f"File [{i_file + 1}/{len(files)}] {file_name} has already been processed, skipping...")
            return

        print(f"{file_str} {source_path} -> {target_path}")

        # Open the files
        with uproot.open(source_path) as source_file:
            # Read the list of trees
            trees = source_file.keys()

            # We need to join the individual cycles
            trees = list(set([tree.split(";")[0] for tree in trees]))
            predictions = {}

            # Loop over all the trees
            for i_tree, tree_name in enumerate(trees):
                tree = source_file[tree_name]
                n_events = tree.num_entries

                print(f"{file_str} {file_name}: [{i_tree + 1}/{len(trees)}] {tree_name} ({n_events} events)")

                # Check if there are no events
                if tree.num_entries == 0:
                    predictions[tree_name] = np.array([])
                    continue

                # Get the branches specified by the model
                predictions_tree = []
                for batch in tqdm(
                        tree.iterate(features, step_size=100),
                        disable=batch_size >= n_events,
                        total=n_events // 100 + 1,
                        file=sys.stdout,
                ):
                    # Get the model's predictions on this tree
                    predictions_tree.append(model(batch))

                predictions[tree_name] = np.concatenate(predictions_tree)

        # Save the predictions
        if include_original:
            # Copy the file
            shutil.copyfile(source_path, target_path)

            # Open with ROOT to update (currently not possible with uproot)
            file = ROOT.TFile.Open(target_path, "UPDATE")

            # Get the list of trees
            trees = file.GetListOfKeys()
            total_trees = trees.GetEntries()

            # Iterate over the trees
            for tree_name, predictions in predictions.items():
                tree_name = tree.GetName()
                tree = file.Get(tree_name)

                # Create the new branch
                prob = array("f", [0])
                bpt = tree.Branch(added_branch_name, prob, f"{added_branch_name}/F")

                # Keep only active branches
                tree.SetBranchStatus("*", 0)
                tree.SetBranchStatus(added_branch_name, 1)

                # Fill the new branch
                for i_event, prediction in enumerate(predictions):
                    tree.GetEntry(i_event)
                    prob[0] = prediction
                    bpt.Fill()

                # Reset the branch status
                tree.SetBranchStatus("*", 1)

                # Write the tree
                tree.Write()

            # Close the file
            file.Close()
        else:
            with uproot.recreate(target_path) as target_file:
                for tree_name, predictions in predictions.items():
                    target_file[tree_name] = {added_branch_name: predictions}

        # Add the file to the list of processed files (create an empty file)
        with open(processed_path, "w") as f:
            pass
    except Exception as e:
        print(f"{file_str} {file_name} failed: {e}")
        failed.append(file_name)

        # Add the file to the list of failed files
        with open(failed_path, "a") as f:
            f.write(file_name + "\n")


for i_file, file_name, source_path, target_path in iterate_files(source_base_dir, target_base_dir, files, restart=args.restart):
    process_file(i_file, file_name, source_path, target_path)


if len(failed) > 0:
    print("Failed files:")
    for file in failed:
        print(file)
