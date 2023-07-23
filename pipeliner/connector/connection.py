import paramiko
from dataclasses import dataclass

from pipeliner.utils import DotDict, green, yellow


@dataclass
class Result:
    success: bool
    stdout: str
    stderr: str
    exit_code: int

    def __iter__(self):
        yield self.success
        yield self.stdout
        yield self.stderr
        yield self.exit_code


class Connection:
    def __init__(self, config: DotDict):
        self.hostname = config.hostname
        self.port = config.port
        self.username = config.username
        self.password = config.password
        self.root = config.root

        # Create an SSH client
        self.client = paramiko.SSHClient()

        # Automatically add the server's host key
        self.client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        self.is_connected = False

    def open(self):
        if self.is_connected:
            print(green(f"Already connected to {self.username}@{self.hostname}:{self.port}"))
            return True

        try:
            print(f"Connecting to {self.username}@{self.hostname}:{self.port}")
            self.client.connect(self.hostname, self.port, self.username, self.password)
            self.is_connected = True
            print(green(f"Connected to {self.username}@{self.hostname}:{self.port}"))
        except Exception as e:
            print(yellow(e))

        return self.is_connected

    def run_command(self, command: str):
        # Run the command, get stdin, stdout, stderr
        _, stdout, stderr = self.client.exec_command(command)

        # Get the exit code
        exit_code = stdout.channel.recv_exit_status()

        # Read the output
        stdout = stdout.read().decode("utf-8")
        stderr = stderr.read().decode("utf-8")

        success = exit_code == 0

        return Result(success, stdout, stderr, exit_code)

    def close(self):
        self.client.close()
        if self.is_connected:
            print(f"Disconnected from {self.username}@{self.hostname}:{self.port}")
        self.is_connected = False

    def __enter__(self):
        if not self.is_connected:
            self.open()

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass
