import autorootcwd  # Do not delete - adds the root of the project to the path

from difflib import get_close_matches
import uproot
import yaml
import os
import numpy as np
from data_processing.processing import ConfigParser
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("mode", help="Mode to run in", choices=["check", "merge", "explain"])
parser.add_argument("--ntuple", help="Path to the ntuple file to check the features",
                    default="~/nominal/mc16a/p4498/346343.root")

args = parser.parse_args()

configs = ["j_bohm", "m_rames", "n_bruscino", "s_konig", "old"]
merged_path = os.path.join(os.path.dirname(__file__), "merged.yaml")

if args.mode == "check":
    sample_file = args.ntuple
    with uproot.open(f"{sample_file}:nominal") as nominal:
        # Skip files with no events
        if nominal.num_entries == 0:
            raise RuntimeError(f"{sample_file} has no events!")

        # Test each yaml file in this directory
        # Check if all the features are present and in the same case
        # Compare in lowercase using the Levenshtein distance
        for config in configs:
            print(f"\n{(' ' + config + ' '):-^40s}")

            config_path = os.path.join(os.path.dirname(__file__), f"{config}.yaml")

            # Read the config file
            with open(config_path, "r") as f:
                features = yaml.safe_load(f)

            present_features = [f for f in nominal.keys()]
            lower_features = [p.lower() for p in present_features]

            n_found = 0
            n_almost_found = 0
            for ft in features:
                if ft in present_features:
                    n_found += 1
                    continue

                if ft.lower() in lower_features:
                    matching = [f for f in present_features if f.lower() == ft.lower()][0]
                    print(f"Feature {ft} found as {matching}")
                    n_almost_found += 1

                else:
                    print(f"Feature {ft} not found")

                    # Find the closest 3 features
                    closest = get_close_matches(ft.lower(), lower_features, n=10, cutoff=0.2)

                    print(f"Closest features: {[[f for f in present_features if f.lower() == c][0] for c in closest]}")
                    print()

            print(f"Found {n_found} features")
            print(f"Found {n_almost_found} features with different case")
            print(f"Found {len(features) - n_found - n_almost_found} features not found")

        print(f"\n{(' Checking categorical features '):-^40s}")

        # Check the categorical features - for each feature print the list of unique values
        categorical_config = os.path.join(os.path.dirname(__file__), "categorical.yaml")
        with open(categorical_config, "r") as f:
            categorical = yaml.safe_load(f)

            for feature in categorical:
                unique = np.unique(nominal[feature].array(library="np"))
                print(f"{feature:>30s}: {len(unique)} unique values: {unique}")

elif args.mode == "merge":
    all_features = set()

    for config in configs:
        config_path = os.path.join(os.path.dirname(__file__), f"{config}.yaml")

        # Read the config file
        with open(config_path, "r") as f:
            features = yaml.safe_load(f)

        features = set(features)
        n_previous_features = len(all_features)

        new_feature = set(features) - all_features
        all_features |= features

        if n_previous_features > 0:
            print(f"Found {len(new_feature)} new features in {config}.yaml: {new_feature}")

    print(f"Found {len(all_features)} features in total")

    # Sort the features alphabetically
    all_features = sorted(all_features)

    with open(merged_path, "w") as f:
        yaml.dump(list(all_features), f)

    print(f"Saved to {merged_path}")

elif args.mode == "explain":
    # Open the merged file and throw if it does not exist
    if not os.path.exists(merged_path):
        raise RuntimeError(f"Could not find {merged_path}")

    # Check if the descriptions.yaml exists as well
    descriptions_path = os.path.join(os.path.dirname(__file__), "descriptions.yaml")
    if not os.path.exists(descriptions_path):
        raise RuntimeError(f"Could not find {descriptions_path}")

    # Read the merged file
    with open(merged_path, "r") as f:
        features = yaml.safe_load(f)

    # Read the descriptions file
    with open(descriptions_path, "r") as f:
        descriptions = yaml.safe_load(f)

    new_features = set()
    for feature in features:
        # If feature ends with a [!DIGIT][DIGIT] we need replace the digit with an X
        if feature[-1].isdigit() and not feature[-2].isdigit():
            feature = feature[:-1] + "X"
        new_features.add(feature)

    features = new_features

    missing = set()
    for feature in features:
        print(f"{feature}: {descriptions.get(feature, 'MISSING')}")
        if feature not in descriptions:
            missing.add(feature)

    if missing:
        print()
        print(f"Found {len(missing)} missing descriptions:")
        for m in missing:
            print(m)
