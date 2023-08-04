import ROOT
import os
import shutil
from argparse import ArgumentParser, BooleanOptionalAction

cut_expression = "custTrigMatch_LooseID_FCLooseIso_DLT && (dilep_type > 0 && (lep_ID_0*lep_ID_1)>0) && ((lep_Pt_0>=10e3&&lep_Pt_1>=10e3)&&(fabs(lep_Eta_0)<=2.5&&fabs(lep_Eta_1)<=2.5)&&(( abs(lep_ID_0) == 13 && lep_isMedium_0 && lep_isolationLoose_VarRad_0 && passPLIVTight_0 )||(( abs(lep_ID_0) == 11 && lep_isTightLH_0 && lep_isolationLoose_VarRad_0 && passPLIVTight_0 && lep_ambiguityType_0 == 0 && lep_chargeIDBDTResult_recalc_rel207_tight_0>0.7 ) && ((!(!(lep_Mtrktrk_atConvV_CO_0<0.1&&lep_Mtrktrk_atConvV_CO_0>=0&&lep_RadiusCO_0>20)&&(lep_Mtrktrk_atPV_CO_0<0.1&&lep_Mtrktrk_atPV_CO_0>=0)))&&!(lep_Mtrktrk_atConvV_CO_0<0.1&&lep_Mtrktrk_atConvV_CO_0>=0&&lep_RadiusCO_0>20))))&& (( abs(lep_ID_1) == 13 && lep_isMedium_1 && lep_isolationLoose_VarRad_1 && passPLIVTight_1 ) || (( abs(lep_ID_1) == 11 && lep_isTightLH_1 && lep_isolationLoose_VarRad_1 && passPLIVTight_1 && lep_ambiguityType_1 == 0 && lep_chargeIDBDTResult_recalc_rel207_tight_1>0.7 )&&((!(!(lep_Mtrktrk_atConvV_CO_1<0.1&&lep_Mtrktrk_atConvV_CO_1>=0&&lep_RadiusCO_1>20)&&(lep_Mtrktrk_atPV_CO_1<0.1&&lep_Mtrktrk_atPV_CO_1>=0)))&&!(lep_Mtrktrk_atConvV_CO_1<0.1&&lep_Mtrktrk_atConvV_CO_1>=0&&lep_RadiusCO_1>20))))) && nTaus_OR==1 && nJets_OR_DL1r_85>=1 && nJets_OR>=4 && ((dilep_type==2) || abs(Mll01-91.2e3)>10e3)"

base_path = "/eos/atlas/atlascerngroupdisk/phys-higgs/HSG8/multilepton_ttWttH/v08/v0801/systematics-full"
target_base_dir = "./ntuples-small"

# Get the subdir as the first argument
parser = ArgumentParser()
parser.add_argument("subdir", help="Subdirectory to process")
parser.add_argument("--restart", action=BooleanOptionalAction, help="Restart from the beginning")

# Parse the arguments
args = parser.parse_args()
subdir = args.subdir
restart = args.restart

source_subdir = os.path.join(base_path, subdir)
target_subdir = os.path.join(target_base_dir, subdir)

# Check if the source and target dirs exist
if not os.path.isdir(base_path):
    exit(f"Source base dir {base_path} does not exist")

# If the restart flag is set, remove the target subdir
if restart and os.path.isdir(target_subdir):
    print(f"Removing target subdir {target_subdir} due to --restart flag")
    shutil.rmtree(target_subdir)

# Create the target dir if it doesn't exist
if not os.path.isdir(target_base_dir):
    os.mkdir(target_base_dir)

# Check if the subdir exists
if not os.path.isdir(source_subdir):
    exit(f"Subdir {subdir} does not exist")

# Create the target subdir if it doesn't exist
if not os.path.isdir(target_subdir):
    os.mkdir(target_subdir)

# Get all the files in the subdir
files = []
remaining = [source_subdir]
while remaining:
    current_dir = remaining.pop()

    for file in os.listdir(current_dir):
        file = os.path.join(current_dir, file)
        if file.endswith(".root"):
            files.append(file)
        elif os.path.isdir(file):
            remaining.append(file)

n_all_files = len(files)
print(f"{len(files)} files found in {source_subdir}")

# Check if the .processed_files file exists
processed_files_path = os.path.join(target_subdir, ".processed_files")
if os.path.isfile(processed_files_path):
    # Read the processed files (by lines)
    with open(processed_files_path, "r") as processed_files_file:
        processed_files = processed_files_file.read().splitlines()
else:
    processed_files = []

print(f"{len(processed_files)} files already processed")

# Checked if the .skipped_files file exists
skipped_files_path = os.path.join(target_subdir, ".skipped_files")
if os.path.isfile(skipped_files_path):
    # Read the skipped files (by lines)
    with open(skipped_files_path, "r") as skipped_files_file:
        skipped_files = skipped_files_file.read().splitlines()
else:
    skipped_files = []

# Remove the processed files and skipped files from the list of files
files = [file for file in files if (file not in processed_files and file not in skipped_files)]

print(f"{len(files)} files remaining")

# Recreate the .processed_files file, and add the processed files once they are done
with open(processed_files_path, "w") as processed_files_file:
    # Write the processed files (by lines)
    processed_files_file.write("\n".join(processed_files))

i_current = n_all_files - len(files)
for source_path in files:
    try:
        target_path = source_path.split(base_path)[1]
        target_path = target_base_dir + target_path

        print(f"File {i_current} of {n_all_files}: {source_path} -> {target_path}")

        # Create the target dir if it doesn't exist
        target_dir = target_path[:target_path.rfind("/")]
        ROOT.gSystem.mkdir(target_dir, True)

        # Open the source file
        source_file = ROOT.TFile.Open(source_path, "READ")
        # And the target file
        target_file = ROOT.TFile.Open(target_path, "RECREATE")

        # Get the list of trees
        trees = source_file.GetListOfKeys()
        total_trees = trees.GetEntries()

        # Iterate over the trees
        selected_none = True
        for i_tree, tree in enumerate(trees):
            tree_name = tree.GetName()
            tree = source_file.Get(tree_name)
            num_events = tree.GetEntries()
            print(f"\tTree {i_tree + 1} of {total_trees}: {tree_name} ({num_events} events)")

            if num_events == 0:
                print("\t\tSkipping empty tree")
                continue

            # Create a new tree with the cut applied
            new_tree = tree.CopyTree(cut_expression)

            # Skip if the new tree is empty
            n_selected_events = new_tree.GetEntries()
            if n_selected_events == 0:
                print("\t\tSkipping empty tree (after selection)")
                continue

            selected_none = False
            print(f"\t\t{n_selected_events} events selected")

            # Write the new tree to the target file
            target_file.cd()
            new_tree.Write()

        # Close the files
        source_file.Close()
        target_file.Close()

        # Remove the target file if no events were selected
        if selected_none:
            print(f"\tRemoving {target_path} as no events were selected")
            os.remove(target_path)

        # Add the file to the list of processed files
        with open(processed_files_path, "a") as processed_files_file:
            processed_files_file.write("\n" + source_path)
    except Exception as e:
        print(f"\tError processing file: {e}")
        # Add the file to the list of skipped files
        with open(skipped_files_path, "a") as skipped_files_file:
            skipped_files_file.write("\n" + source_path)
