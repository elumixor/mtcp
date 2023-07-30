import autorootcwd  # Do not delete - adds the root of the project to the path

# Load and test it
from data_processing.processing import load_data


def check(output_path: str):
    data = load_data(output_path)

    total_raw = 0
    total_weighted = 0
    total_selected_raw = 0
    total_selected_weighted = 0

    for y, y_name in enumerate(data.y_names):
        w = data.w[data.y == y]
        num_raw = w.shape[0]
        num_weighted = w.sum()

        selected_w = data.w[(data.y == y) & data.selected]
        num_selected_raw = selected_w.shape[0]
        num_selected_weighted = selected_w.sum()

        print(f"{y_name:>10s}: {num_raw:>10d} raw {num_weighted:>12.4f} weighted {num_selected_raw:>10d} raw {num_selected_weighted:>12.4f} weighted")

        total_raw += num_raw
        total_weighted += num_weighted
        total_selected_raw += num_selected_raw
        total_selected_weighted += num_selected_weighted

    print(f"{'Total':>10s}: {total_raw:>10d} raw {total_weighted:>12.4f} weighted {total_selected_raw:>10d} raw {total_selected_weighted:>12.4f} weighted")


if __name__ == "__main__":
    check("data_processing/output")
