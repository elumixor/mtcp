import paramiko
from dataclasses import dataclass
import json

from pipeliner.utils import DotDict, green, yellow, red, cyan, orange


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
    def __init__(self, cluster: str, config: DotDict):
        self.cluster = cluster
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

    @property
    def exports(self):
        return f"export MTCP_ROOT={self.root}/mtcp && \\\n" + \
            f"export MTCP_ARTIFACTS_DIR=$MTCP_ROOT/artifacts && \\\n" + \
            f"export MTCP_JOBS_DIR=$MTCP_ROOT/jobs && \\\n" + \
            f"export MTCP_TREX_DIR=$MTCP_ROOT/trex-fitter && \\\n" + \
            f"export MTCP_CLUSTER={self.cluster} && \\\n"

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

    def run_command(self, command: str, root=None, debug=False):
        if not self.is_connected:
            self.open()

        if root is None:
            root = self.root

        root_cmd = f"cd {root} && \\\n" if root != "" else ""

        # Run the command, get stdin, stdout, stderr
        command = f"{self.exports}{root_cmd}{command}"
        _, stdout, stderr = self.client.exec_command(command)

        # Get the exit code
        exit_code = stdout.channel.recv_exit_status()

        # Read the output
        stdout = stdout.read().decode("utf-8")
        stderr = stderr.read().decode("utf-8")

        success = exit_code == 0

        if debug or not success:
            self.print_result(command, stdout, stderr, exit_code, log_command=True)

        return Result(success, stdout, stderr, exit_code)

    def git_sync(self, debug=False):
        with self:
            self.log(f"Syncing git repo")

            # First, check if the directory exists and is okay
            result = self.run_command("cd $MTCP_ROOT && git status", debug=debug)
            if not result.success:
                # If it doesn't, clone the repo
                self.log(red("Repo does not exist or is corrupted."), orange("Performing clean clone..."))
                result = self.run_command("rm -rf $MTCP_ROOT && \\\n" +
                                          "git clone git@github.com:elumixor/mtcp.git $MTCP_ROOT", debug=debug)
                if not result.success:
                    raise Exception(f"Failed to clone the repo.\n{result.stderr}")

            # At this point we have the repo, but we are not sure if we are on the master branch
            # or if we have any uncommitted changes.

            # First, let's fetch the latest changes from the remote
            self.log("Fetching latest changes...")
            result = self.run_command("cd $MTCP_ROOT && git fetch origin", debug=debug)
            if not result.success:
                raise Exception(f"Failed to fetch latest changes.\n{result.stderr}")

            # If we are NOT on the master branch...
            result = self.run_command("cd $MTCP_ROOT && git rev-parse --abbrev-ref HEAD", debug=debug)
            if not result.success:
                raise Exception(f"Failed to get current branch.\n{result.stderr}")

            if result.stdout.strip() != "master":
                self.log(f"Branch: {cyan(result.stdout.strip())}")

                # Try to merge the master branch. Abort if there are conflicts.
                result = self.run_command(
                    "cd $MTCP_ROOT && \\\n" +
                    "git merge origin/master", debug=debug)

                # If exit status is 0, then merge was successful and we have the latest changes from the master branch
                # while keeping the changes on the current branch. Then we can push the changes to the remote. And
                # complete the sync.
                if result.success:
                    self.log(green("Successfully merged changes from master."))
                    result = self.run_command(
                        "cd $MTCP_ROOT && \\\n" +
                        "git push -u origin $MTCP_GIT_CLUSTER_BRANCH", debug=debug)
                    if not result.success:
                        raise Exception(f"Failed to push changes to remote.\n{result.stderr}")

                else:
                    # Otherwise, we should abort the merge and leave all as is
                    self.log(red("Failed to merge changes from master. Aborting..."))

                    merge_result = result
                    result = self.run_command(
                        "cd $MTCP_ROOT && \\\n" +
                        "git merge --abort", debug=debug)
                    if not result.success:
                        raise Exception(f"Failed to abort merge.\n{result.stderr}")

                    raise Exception(f"Failed to merge changes from master.\n{merge_result.stderr}")
            else:
                self.log(f"Branch: {green(result.stdout.strip())}")

                # We are on the master branch!
                # Just pull the changes. If failed to do so - then local changes would be overwritten.
                result = self.run_command(
                    "cd $MTCP_ROOT && \\\n" +
                    "git pull", debug=debug)

                if not result.success:
                    self.log(red("Failed to pull changes."))
                    raise Exception(f"Failed to pull changes.\n{result.stderr}")

            # All should be fine by now. We can check the status of the repo.
            result = self.run_command("cd $MTCP_ROOT && git status", debug=debug)
            return dict(status=result.stdout.strip())

    def close(self):
        self.client.close()
        if self.is_connected:
            self.log(f"Disconnected from {self.username}@{self.hostname}:{self.port}")
        self.is_connected = False

    def log(self, *args, **kwargs):
        print(orange(f"[{self.cluster}]"), *args, **kwargs)

    def print_result(self, cmd: str, stdout: str, stderr: str, exit_code, log_command=False):
        # Print command
        if log_command:
            print(f"\n{' Command ':=^30}\n")
            print(cmd)

        # Print exit code
        print(f"\nExit Code: {green('0') if exit_code == 0 else red(exit_code)}")

        if stdout or stderr:
            print()

        if stdout:
            print(f"{' stdout ':=^30}\n")
            print(stdout)
            print()

        if stderr:
            print(f"{' stderr ':=^30}\n")
            print(stderr)
            print()

    def __enter__(self):
        if not self.is_connected:
            self.open()

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass
