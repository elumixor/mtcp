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
from pipeliner.utils import read_yaml

parser = ArgumentParser()
parser.add_argument("--config", default="./friend_ntuples/config.yaml", help="Path to the config file")
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
    config = read_yaml(args.config)
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
model_path = os.path.abspath(model_path)
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
target_base_dir = args.target_base_dir if args.target_base_dir else os.path.join(config.target_base_dir, "friend")
files = [args.file] if args.file else config.files if "files" in config else get_all_files(source_base_dir)
for i_file, source_path, target_path, processed_path in iterate_files(source_base_dir, target_base_dir, files, restart=args.restart):
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

            print(f"\tTree [{i_tree + 1}/{len(trees)}] {tree_name} ({n_events} events)")

            # Check if there are no events
            if tree.num_entries == 0:
                print(f"\t\t{tree_name} has no events")
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