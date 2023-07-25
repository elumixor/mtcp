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

    @property
    def exports(self):
        return f"export MTCP_JOB={self.name} && \\\n" + \
            f"export MTCP_JOB_DIR=$MTCP_JOBS_DIR/$MTCP_JOB && \\\n" + \
            f"export MTCP_JOB_ARTIFACTS_DIR=$MTCP_JOB_DIR/artifacts && \\\n" + \
            f"export MTCP_JOB_LOGS_DIR=$MTCP_JOB_DIR/logs"

    @property
    def json(self):
        return dict(
            name=self.name,
            description=self.description,
            clusters=self.allowed_clusters,
            command=self.command,
            artifacts=self.artifacts,
        )

    def log(self, *args, cluster=None, error=False, **kwargs):
        log = self.connector[cluster].log if cluster else print
        log(magenta(f"[{self.name}]"), "" if not error else red("[ERROR]"), *args, **kwargs)

    def log_condor_status(self, cluster: str, status: dict):
        if "condor" not in status:
            return

        condor_status = status["condor"]
        id, status_str = condor_status["id"], condor_status["status"]

        color = orange if status_str == "running" else \
            green if status_str == "done" else \
            red if status_str == "interrupted" \
            else cyan

        status_str = color(status_str)

        self.log(f"Job ID: {cyan(id)}. Status: {status_str}", cluster=cluster)

    def run_command(self, command: str, cluster: str, debug=False):
        result = self.connector[cluster].run_command(f"{self.exports} && \\\n{command}", debug=debug)
        self.log(green("Success") if result.success else red("Failed"), cluster=cluster, error=not result.success)
        return result

    def run(self, cluster: str, clean=True, debug=False):
        if clean:
            self.delete_artifacts(cluster)

        self.log("Running the job...", cluster=cluster)

        use_condor = self.condor and cluster == "cern"
        if use_condor:
            self.log("Submitting to condor", cluster=cluster)
            condor_cmd = " python3 $MTCP_ROOT/pipeliner/condor/submit.py"
        else:
            condor_cmd = ""

        result = self.run_command(f"{condor_cmd} {self.command}", cluster, debug=debug)

        if use_condor and result.success:
            status = json.loads(result.stdout)
            self.log_condor_status(cluster, status)
            return status

        return result

    def check_status(self, cluster: str, debug=False):
        self.log("Checking status...", cluster=cluster)

        result = self.run_command(f"python3 $MTCP_ROOT/pipeliner/condor/check_status.py", cluster, debug=debug)

        if result.success:
            status = json.loads(result.stdout)
            self.log_condor_status(cluster, status)

            return status

        return result

    def interrupt(self, cluster: str, debug=False):
        self.log("Interrupting...", cluster=cluster)
        result = self.run_command(f"python3 $MTCP_ROOT/pipeliner/condor/interrupt.py", cluster, debug=debug)
        success, stdout, stderr, exit_code = result
        return json.loads(stdout) if success else dict(error=exit_code, message=stderr)

    def delete_artifacts(self, cluster: str, debug=False):
        results = {}

        artifacts_str = ", ".join([cyan(f"\"{artifact}\"") for artifact in self.artifacts])
        self.log(f"Deleting artifacts: {artifacts_str}", cluster=cluster)

        for artifact in self.artifacts:
            result = self.run_command(f"rm -rf {artifact}", cluster, debug=debug)

        if result.success:
            self.log(green("Deleted") + " " + cyan(f"\"{artifact}\""), cluster=cluster)
            results[artifact] = not result.success

        return results
