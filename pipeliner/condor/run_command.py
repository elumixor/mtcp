import subprocess

def run_command(command):
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    stdout = process.stdout.read().decode("utf-8")
    stderr = process.stderr.read().decode("utf-8")
    exit_code = process.poll()

    if exit_code != 0:
        print(f"Failed: {command}", file=sys.stderr)
        print(stderr, file=sys.stderr)
        exit(1)

    return stdout