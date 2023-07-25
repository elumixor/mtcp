def read_file(file_path: str):
    """Simple utility to read the text with with the characters '%' and '#' as comments."""
    with open(file_path, "r") as f:
        lines = f.readlines()

        # Remove comments starting with % and # unless they appear in a string
        for ch in ["%", "#"]:
            lines = [line.split(ch)[0].rstrip()
                     if ch in line and line.split(ch)[0].count('"') % 2 == 0
                     else line.rstrip()
                     for line in lines]

        lines = [line for line in lines if line.strip() != ""]

        return lines
