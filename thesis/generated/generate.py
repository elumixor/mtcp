import autorootcwd  # Do not delete - adds the root of the project to the path

import numpy as np

# Load the data files

w = np.load("data_processing/output/w.npy")
y = np.load("data_processing/output/y.npy")
y_names = np.load("data_processing/output/y_names.npy")
selected = np.load("data_processing/output/selected.npy")

# For each process and (all)
# Inside SR and ALL
# Raw/weighted

# Example:
# Process | SR (Raw) | SR (Weighted) | ALL (Raw) | ALL (Weighted)
# ---------------------------------------------------------------
# ttH     | ...      | ...           | ...       | ...

# Generate the latex table for the number of events
with open("thesis/generated/num_events.tex", "w") as f:
    f.write("\\begin{tabular}{l|rr|rr}\n")
    f.write("Process & SR (Raw) & SR (Weighted) & ALL (Raw) & ALL (Weighted) \\\\\n")
    f.write("\\hline\n")

    for i, name in enumerate(y_names):
        selected_all = y == i
        all_raw = selected_all.sum()
        all_weighted = (w[selected_all]).sum()

        selected_sr = selected & (y == i)
        sr_raw = selected_sr.sum()
        sr_weighted = (w[selected_sr]).sum()

        f.write(f"{name} & {sr_raw} & {sr_weighted:.2f} & {all_raw} & {all_weighted:.2f} \\\\\n")

    total_raw = y.shape[0]
    total_weighted = w.sum()
    total_selected_raw = selected.sum()
    total_selected_weighted = (w[selected]).sum()

    f.write("\\hline\n")
    f.write(f"Total & {total_selected_raw} & {total_selected_weighted:.2f} & {total_raw} & {total_weighted:.2f} \\\\\n")

    f.write("\\end{tabular}\n")
