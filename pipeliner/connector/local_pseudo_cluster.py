import os
import sys
import subprocess

from pipeliner.remote.download import transfer
from pipeliner.utils import orange

from .result import Result
from .cluster import Cluster

class LocalPseudoCluster:
    def __init__(self):
        self.name = "local"
        # Root is where the current working directory is
        # The autorootcwd ensures that the root is always the same
        self.root = os.path.dirname(os.path.abspath(os.curdir))

    @property
    def exports(self):
        return f"export MTCP_ROOT={self.root}/mtcp && \\\n" + \
            f"export MTCP_ARTIFACTS_DIR=$MTCP_ROOT/artifacts && \\\n" + \
            f"export MTCP_JOBS_DIR=$MTCP_ROOT/jobs && \\\n" + \
            f"export MTCP_TREX_DIR=$MTCP_ROOT/trex-fitter && \\\n" + \
            f"export MTCP_CLUSTER={self.name} && \\\n"

    def log(self, *args, **kwargs):
        print(orange(f"[{self.name}]"), *args, **kwargs)

    def close(self):
        pass

    def run_command(self, command: str, root=None, stdin=None, debug=False, silent=False):
        if root is None:
            root = self.root

        root_cmd = f"cd {root} && \\\n" if root != "" else ""

        # Run the command, get stdin, stdout, stderr
        command = f"{self.exports}{root_cmd}{command}"
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        stdout = process.stdout.read().decode("utf-8")
        stderr = process.stderr.read().decode("utf-8")
        exit_code = process.poll()

        success = exit_code == 0

        if not silent and (not success or debug):
            if not success:
                print(f"Failed: {command}", file=sys.stderr)

            print(stdout, file=sys.stderr)
            print(stderr, file=sys.stderr)

            print(f"Exit code: {exit_code}", file=sys.stderr)

        return Result(success, stdout, stderr, exit_code)


    def get_file(self, file: str, cluster_from: Cluster, exports="", debug=False):
        with cluster_from:
            # Get the file name on the remote cluster
            result = cluster_from.run_command(f"{exports} && \\\n" + \
                                              f"echo {file}", debug=debug)
            file_from = result.stdout.strip()

            if debug:
                self.log(f"Getting {file_from} from {cluster_from.name}")

            # Get the file name on the local cluster
            # Strip the root
            result = self.run_command(f"{exports} && \\\n" + \
                                      f"echo {file}", debug=debug)
            file_to = result.stdout.strip()

            if debug:
                self.log(f"File will be saved to {file_to}")

            # We need to create the containing directory locally if it doesn't exist
            dirname = os.path.dirname(file_to)
            if not os.path.exists(dirname):
                os.makedirs(dirname)

            # We are ready to transfer the file
            transfer(cluster_from.client, file_from, file_to)