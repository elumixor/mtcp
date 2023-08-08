import autorootcwd  # Do not delete - adds the root of the project to the path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data files

# w = np.load("data_processing/output/w.npy")
# y = np.load("data_processing/output/y.npy")
# y_names = np.load("data_processing/output/y_names.npy")
# selected = np.load("data_processing/output/selected.npy")

# # For each process and (all)
# # Inside SR and ALL
# # Raw/weighted

# # Example:
# # Process | SR (Raw) | SR (Weighted) | ALL (Raw) | ALL (Weighted)
# # ---------------------------------------------------------------
# # ttH     | ...      | ...           | ...       | ...

# latex_names = dict(
#     ttH="\\tth",
#     ttW="\\ttw",
#     ttW_EW="\\ttwew",
#     ttZ="\\ttz",
#     ttbar="\\ttbar",
#     VV="$VV$",
#     tZ="$tZ$",
#     WtZ="$WtZ$",
#     tW="$tW$",
#     threeTop="$t\\bar{t}t$",
#     fourTop="$t\\bar{t}t\\bar{t}$",
#     ggVV="$ggVV$",
#     VVV="$VVV$",
#     VH="$VH$",
#     WttW="$WttW$",
#     tHjb="$tHjb$",
#     tWH="$tWH$",
# )

# # Generate the latex table for the number of events
# with open("thesis/generated/num_events.tex", "w") as f:
#     f.write("\\begin{tabular}{l|rr|rr}\n")
#     f.write("Process & SR (Raw) & SR (Weighted) & ALL (Raw) & ALL (Weighted) \\\\\n")
#     f.write("\\hline\n")

#     for i, name in enumerate(y_names):
#         selected_all = y == i
#         all_raw = selected_all.sum()
#         all_weighted = (w[selected_all]).sum()

#         selected_sr = selected & (y == i)
#         sr_raw = selected_sr.sum()
#         sr_weighted = (w[selected_sr]).sum()

#         name = latex_names[name]
#         f.write(f"{name} & {sr_raw} & {sr_weighted:.2f} & {all_raw} & {all_weighted:.2f} \\\\\n")

#     total_raw = y.shape[0]
#     total_weighted = w.sum()
#     total_selected_raw = selected.sum()
#     total_selected_weighted = (w[selected]).sum()

#     f.write("\\hline\n")
#     f.write(f"Total & {total_selected_raw} & {total_selected_weighted:.2f} & {total_raw} & {total_weighted:.2f} \\\\\n")

#     f.write("\\end{tabular}\n")

# Generate the plot with the Poisson pdf for different values of mu
# Then also generate the pdf of the gaussian for same mean
fig, ax = plt.subplots()
mu_values = [0.5, 1, 2, 3, 5, 10]
x = np.arange(0, 20)

for mu in mu_values:
    f = [np.math.factorial(i) for i in x]
    y = np.exp(-mu) * mu ** x / f
    ax.plot(x, y, label=f"$\\lambda={mu}$")

# Plot the gaussian
mu = max(mu_values)
sigma = np.sqrt(mu)
y = 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

ax.plot(x, y, label=f"Gaussian ($\\mu={mu}$, $\\sigma={sigma:.2f}$)", linestyle="--", color="black")

ax.set_xticks(x)
ax.set_xlabel("Number of events")
ax.set_ylabel("Probability")
ax.set_title("Poisson distribution")
ax.legend()
fig.savefig("thesis/generated/poisson.pdf")
