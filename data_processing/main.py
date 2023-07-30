import autorootcwd  # Do not delete - adds the root of the project to the path

from data_processing.processing import read_region, process_data, save_data, ConfigParser, read_yaml
from data_processing.check import check


# Read the config file
config = read_yaml("config")

# # Create the trex config parser
# parser = ConfigParser(config["trex_config"])

# training_features = read_yaml(config["features"])
# print(f"Using the following {len(training_features)} features:")
# for f in training_features:
#     print(f"- {f}")

# categorical = read_yaml(config["categorical_features"])
# invalid_values = read_yaml(config["invalid_values"])
# array_features = read_yaml(config["array_features"])
# categorical = [f for f in categorical if f in training_features]
# invalid = {key: value for key, value in invalid_values.items() if key in training_features}
# array_features = [f for f in array_features if f in training_features]

# # Open all the files and read data into awkward array. Also read the weight. Then convert awkward arrays to numpy
# data = read_region(
#     config["region"],
#     parser,
#     training_features,
#     array_features,
#     nested_size=config["n_jets"],
#     samples=config["samples"],
# )

# data = process_data(data, categorical, invalid)

# # Save all of it
# save_data(data, config["output_path"])

# Check if all went well
check(config["output_path"])
