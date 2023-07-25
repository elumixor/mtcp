import os


def get_all_files(base_dir: str):
    # Get all the files in the subdir
    files = []
    remaining = [base_dir]
    while remaining:
        current_dir = remaining.pop()

        for file in os.listdir(current_dir):
            file = os.path.join(current_dir, file)
            if file.endswith(".root"):
                files.append(file)
            elif os.path.isdir(file):
                remaining.append(file)

    return files
