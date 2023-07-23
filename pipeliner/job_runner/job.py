import json
from pipeliner.connector import Connector
from pipeliner.utils import inject, magenta, orange, red, green, cyan
from connector import Result

from dataclasses import dataclass
from connector import Result


@dataclass
class CondorResult(Result):
    condor_ids: str

    def __iter__(self):
        yield from super().__iter__()
        yield self.condor_ids


@dataclass
class JobStatus:
    status: str
    log: str | None = None


class Job:
    def __init__(self, job_name, job_config):
        self.name = job_name
        self.description = job_config.description
        self.allowed_clusters = job_config.clusters
        self.command = job_config.command
        self.artifacts = job_config.artifacts
        self.condor = True if "condor" in job_config and job_config.condor else False
        self.connector = inject(Connector)

    def __repr__(self):
        return f"Job(\"{self.name}\", clusters={self.allowed_clusters})"

    def json(self):
        return {
            "name": self.name,
            "description": self.description,
            "clusters": self.allowed_clusters,
            "command": self.command,
            "artifacts": self.artifacts,
        }

    def log(self, *args, cluster=None, error=False, **kwargs):
        if cluster:
            s = orange(f"[{cluster}]") + "::" + magenta(f"[{self.name}]")
        else:
            s = magenta(f"[{self.name}]")

        if error:
            s += red(" [ERROR]")

        return print(s, *args, **kwargs)

    def log_output(self, cmd, stdout, stderr, debug=False):
        if debug:
            print("\n\t=== COMMAND ===\n")
            print(cmd)

        # Add ">" to the beginning of each line
        stderr = "\n".join(["> " + line for line in stderr.strip().split("\n")]) if stderr else ""
        stdout = "\n".join(["> " + line for line in stdout.strip().split("\n")]) if stdout else ""

        if stdout or stderr:
            print()

        if stdout:
            print("\t=== STDOUT ===\n")
            print(stdout)
            print()

        if stderr:
            print("\t=== STDERR ===\n")
            print(stderr)
            print()

    def log_condor_status(self, cluster: str, status: dict):
        if "condor" not in status:
            return

        status = status["condor"]
        id, status_str = status["id"], status["status"]
        color = orange if status_str == "running" else \
            green if status_str == "done" else \
            red if status_str == "interrupted" \
            else cyan
        status_str = color(status_str)
        self.log(f"Job ID: {cyan(id)}. Status: {status_str}", cluster=cluster)

    def get_exports(self, cluster: str, root: str):
        return f"export MTCP_ROOT={root}/mtcp" + \
            f" && export MTCP_ARTIFACTS_DIR=$MTCP_ROOT/artifacts" + \
            f" && export MTCP_JOBS=$MTCP_ROOT/jobs" + \
            f" && export MTCP_CLUSTER={cluster}" +     \
            f" && export MTCP_JOB={self.name}" + \
            f" && export MTCP_JOB_DIR=$MTCP_JOBS/$MTCP_JOB" + \
            f" && export MTCP_JOB_ARTIFACTS_DIR=$MTCP_JOB_DIR/artifacts" + \
            f" && export MTCP_JOB_LOGS_DIR=$MTCP_JOB_DIR/logs"

    def run(self, cluster: str, clean=True, debug=False):
        if clean:
            self.delete_artifacts(cluster)

        with self.connector[cluster] as connection:
            self.log("Running the job...", cluster=cluster)

            root = connection.root

            use_condor = self.condor and cluster == "cern"
            if use_condor:
                self.log("Submitting to condor", cluster=cluster)
                cmd = f"cd {root} && {self.get_exports(cluster, root)} && python3 $MTCP_ROOT/pipeliner/condor/submit.py {self.command}"
            else:
                cmd = f"cd {root} && {self.get_exports(cluster, root)} && {self.command}"

            result = connection.run_command(cmd)

            self.log(green("Success") if result.success else red("Failed"), cluster=cluster, error=not result.success)

            if debug or not result.success:
                self.log_output(cmd, result.stdout, result.stderr, debug=debug)

            if use_condor and result.success:
                status = json.loads(result.stdout)
                self.log_condor_status(cluster, status)
                return status

            return result

    def check_artifacts(self, cluster: str):
        results = {}

        with self.connector[cluster] as connection:
            self.log("Checking artifacts...", cluster=cluster)

            root = connection.root

            for artifact in self.artifacts:
                cmd = f"cd {root} && {self.get_exports(cluster, root)} && ls {artifact}"
                result = connection.run_command(cmd)

                results[artifact] = result.success

                self.log("Artifact " + cyan(f"\"{artifact}\" ") +
                         (green("exists") if result.success else red("does not exist")),
                         cluster=cluster)

        return results

    def check_status(self, cluster: str, debug=False):
        with self.connector[cluster] as connection:
            self.log("Checking status...", cluster=cluster)

            root = connection.root

            cmd = f"cd {root} && {self.get_exports(cluster, root)} && python3 $MTCP_ROOT/pipeliner/condor/check_status.py"
            result = connection.run_command(cmd)

            if debug or not result.success:
                self.log_output(cmd, result.stdout, result.stderr, debug=debug)

            if result.success:
                status = json.loads(result.stdout)
                self.log_condor_status(cluster, status)

                return status
            else:
                error = result.exit_code
                message = result.stderr
                raise Exception(f"Error checking status: {error}. Stderr:\n{message}")

    def interrupt(self, cluster: str, debug=False):
        with self.connector[cluster] as connection:
            self.log("Interrupting...", cluster=cluster)

            root = connection.root

            cmd = f"cd {root} && {self.get_exports(cluster, root)} && python3 $MTCP_ROOT/pipeliner/condor/interrupt.py"
            result = connection.run_command(cmd)

            self.log(green("Success") if result.success else red("Failed"), cluster=cluster, error=not result.success)

            if debug or not result.success:
                self.log_output(cmd, result.stdout, result.stderr, debug=debug)

            return json.loads(result.stdout) if result.success else {"error": result.exit_code, "message": result.stderr}

    def delete_artifacts(self, cluster: str, debug=False):
        results = {}
        with self.connector[cluster] as connection:

            artifacts = self.artifacts
            root = connection.root

            artifacts_str = ", ".join([cyan(f"\"{artifact}\"") for artifact in artifacts])
            self.log(f"Deleting artifacts: {artifacts_str}", cluster=cluster)

            for artifact in artifacts:
                cmd = f"cd {root} && {self.get_exports(cluster, root)} && rm -rf {artifact}"

                result = connection.run_command(cmd)

            if debug or not result.success:
                self.log_output(cmd, result.stdout, result.stderr, debug=debug)
            else:
                self.log(green("Deleted") + " " + cyan(f"\"{artifact}\""), cluster=cluster)

                results[artifact] = result.success

        return results
