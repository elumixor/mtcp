import os
import re
import yaml

from .parsing import fix_syntax
from .read_file import read_file


class ConfigParser:
    def __init__(self, config_path: str):
        super().__init__()

        # This file is basically a yaml file but with % as an additional comment character, and INCLUDE commands

        base_dir = "trex-fitter"

        # Read the file
        lines = read_file(os.path.join(base_dir, config_path))

        # While there are INCLUDE commands, replace them with the content of the file
        # We need a recursive function that will track the current dir as each next include is relative to the previous one
        def replace_includes(lines, current_dir):
            new_lines = []
            for line in lines:
                if line.startswith("INCLUDE"):
                    include_path = line.split()[1]
                    include_path = os.path.join(current_dir, include_path)
                    include_lines = read_file(include_path)
                    new_lines += replace_includes(include_lines, os.path.dirname(include_path))
                else:
                    new_lines.append(line)
            return new_lines
        lines = replace_includes(lines, os.path.dirname(os.path.join(base_dir, config_path)))

        # Another incompatibility with yaml is that major blocks have their name specified in the same line
        # We need to split the lines so that each block is on a separate line
        new_lines = []
        for line in lines:
            # If there is no indent
            if line == line.lstrip():
                # Then this is a major block. The name is after the colon
                block_type, block_name = [i.strip() for i in line.split(":")]

                # Add the block type and name to the line
                new_lines.append(f"{block_type}:")
                new_lines.append(f"  BLOCK_NAME: {block_name}")
            else:
                new_lines.append(line)
        lines = new_lines

        # Also, sometimes there is a list of values, but it is not wrapped with []
        # We need to add the brackets
        new_lines = []
        for line in lines:
            # If there is no indent - skip
            if line == line.lstrip():
                new_lines.append(f"- {line}")
                continue

            # Otherwise, check for coma-separated values
            colon_index = line.find(":")
            key = line[:colon_index].strip()
            value = line[colon_index + 1:].strip()

            if "," in value:
                value = "[" + value + "]"
            else:
                # Otherwise check if value is wrapped with ""
                if not value.startswith('"') or not value.endswith('"'):
                    value = f"\"{value}\""

            new_lines.append(f"    {key}: {value}")
        lines = new_lines

        # For the \ in strings, we need to add an extra \ to escape it
        lines = [line.replace("\\", "\\\\") for line in lines]

        yaml_str = "\n".join(lines)
        data = yaml.safe_load(yaml_str)

        self.job = [i for i in data if "Job" in i][0]["Job"]
        self.samples = {i["Sample"]["BLOCK_NAME"]: i["Sample"] for i in data if "Sample" in i.keys()}
        self.regions = {i["Region"]["BLOCK_NAME"]: i["Region"] for i in data if "Region" in i.keys()}

        # Get the replacement file
        replacement_file = os.path.join(base_dir, self.job["ReplacementFile"])
        replacements_lines = read_file(replacement_file)
        self.replacements = {}

        # Parse the replacements file
        for line in replacements_lines:
            colon_index = line.find(":")
            key = line[:colon_index].strip()
            value = line[colon_index + 1:].strip()

            self.replacements[key] = value

    @property
    def ntuple_name(self):
        return self._subs(self.job["NtupleName"])

    @property
    def ntuple_base_path(self):
        return self._subs(self.job["NtuplePaths"])

    @property
    def luminosity(self):
        return float(self._subs(self.job["Lumi"]))

    def cut_features(self, region: str):
        """Returns a list of all the features used in the cut expression of the given region"""
        # Get the cut expression for the region
        cut_expr = self.cut_expr(region)

        # Select all the features that are used in the cut expression
        features = re.findall(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\b(?!\s*\()", cut_expr)

        # Remove duplicates
        features = list(set(features))

        return features

    def weight_expr(self, process=None, with_luminosity=False):
        # return "nJets_OR"

        weight = self._subs(self.job["MCweight"])
        # weight = "(" + weight.replace("&&", "&").replace("||", "|").replace("&", ") & (").replace("|", ") | (").replace("!", "~") + ")"

        if process is not None:
            process_config = self.samples[process]
            if "MCweight" in process_config:
                process_weight = self._subs(process_config["MCweight"])
                weight = f"({weight}) * ({process_weight})"

        if with_luminosity:
            weight = f"({self.luminosity}) * ({weight})"

        weight = fix_syntax(weight)

        return weight

    def files_by_process(self, full_path=True):
        files = dict()

        for process, sample in self.samples.items():
            paths = sample["NtupleFiles"]
            paths = str(paths)
            paths = self._subs(paths)
            paths = paths.split(",")

            if full_path:
                paths = [os.path.join(self.ntuple_base_path, path.strip()) for path in paths]
                paths = [f"{path}.root" for path in paths]

            files[process] = paths

        return files

    def cut_expr(self, region_name: str, sample=None):
        if region_name not in self.regions:
            raise Exception(f"Unknown region: {region_name}")

        region = self.regions[region_name]
        selection = region["Selection"]
        selection = self._subs(selection)

        if sample is not None:
            sample_config = self.samples[sample]
            if "Selection" in sample_config:
                print(f"Using selection from {sample} sample")
                sample_selection = sample_config["Selection"]
                sample_selection = self._subs(sample_selection)
                selection = f"({selection}) && ({sample_selection})"
                # selection = sample_selection

        # First, we need to correctly handle "!". To do so we need to add "(" after it and ")" before the matching ")"
        # i = 0
        # while i < len(selection):
        #     char = selection[i]
        #     if char == "!":
        #         num_brackets = 0
        #         for j, char2 in enumerate(selection[i + 1:]):
        #             if char2 == "(":
        #                 num_brackets += 1
        #             if char2 == ")":
        #                 num_brackets -= 1

        #             if num_brackets == 0:
        #                 selection = selection[:i + j + 2] + ")" + selection[i + j + 2:]  # Add the closing bracket
        #                 selection = selection[:i + 1] + "(" + selection[i + 1:]  # Add the opening bracket
        #                 break
        #     i += 1

        # We need to replace "&&" with "&" and "||" with "|", however we need to keep the order of the operations
        # selection = "(" + selection.replace("&&", ") & (").replace("||", ") | (").replace("!", "~") + ")"

        selection = fix_syntax(selection)

        return selection

    def _subs(self, string):
        # Replace while string contains a substring starting with "XXX"
        while any([key in string for key in self.replacements]):
            for key, val in self.replacements.items():
                string = string.replace(key, val)

        # If it still has a "XXX" in it, it's an unknown placeholder
        if "XXX" in string:
            raise Exception(f"Unknown placeholder in string: {string}")

        return string
