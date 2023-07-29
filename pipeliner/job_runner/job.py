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
        self.config = job_config

        self.description = job_config.description
        self.allowed_clusters = job_config.clusters
        self.command = job_config.command
        self.artifacts = job_config.artifacts

        if "condor" in job_config:
            if isinstance(job_config.condor, bool):
                self.condor = dict(used=True, params={}) if job_config.condor else dict(used=False)
            else:
                self.condor = dict(used=True, params=job_config.condor)
        else:
            self.condor = dict(used=False)

        self.connector = inject(Connector)
        self.statuses: dict | None = None

    def __repr__(self):
        return f"Job(\"{self.name}\", clusters={self.allowed_clusters})"

    @property
    def exports(self):
        s = f"export MTCP_JOB={self.name} && \\\n" + \
            f"export MTCP_JOB_DIR=$MTCP_JOBS_DIR/$MTCP_JOB && \\\n" + \
            f"export MTCP_JOB_ARTIFACTS_DIR=$MTCP_JOB_DIR/artifacts && \\\n" + \
            f"export MTCP_JOB_LOGS_DIR=$MTCP_JOB_DIR/logs"

        if "require_memory" in self.config:
            s += f" && \\\n" + \
                f"export MTCP_REQUIRE_MEMORY={self.config.require_memory}"

        return s

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
            magenta if status_str == "interrupted" \
            else cyan

        status_str = color(status_str)

        self.log(f"Job ID: {cyan(id)}. Status: {status_str}", cluster=cluster)

    def run_command(self, command: str, cluster: str, debug=False, silent=False):
        return self.connector[cluster].run_command(f"{self.exports} && \\\n" +
                                                   command, debug=debug, silent=silent)

    def run(self, cluster: str, clean=True, debug=False):
        self.statuses = None

        if clean:
            self.delete_artifacts(cluster)

        self.log("Running the job...", cluster=cluster)

        # Add command to make artifacts directories if they don't exist
        mkdir_cmd = "mkdir -p $MTCP_JOB_ARTIFACTS_DIR && \\\n" + \
                    "mkdir -p $MTCP_ARTIFACTS_DIR && \\\n" + \
                    "[ ! -f $MTCP_JOB_LOGS_DIR ] && mkdir -p $MTCP_JOB_LOGS_DIR || : && \\\n"

        use_condor = self.condor['used'] and cluster == "cern"
        if use_condor:
            self.log("Submitting to condor", cluster=cluster)
            condor_cmd = "python3 $MTCP_ROOT/pipeliner/remote/submit.py \\\n"
            for key, value in self.condor["params"].items():
                condor_cmd += f"--{key} {value} \\\n"
        else:
            condor_cmd = ""

        result = self.run_command(f"{mkdir_cmd}{condor_cmd}{self.command}", cluster, debug=debug)

        if use_condor and result.success:
            status = json.loads(result.stdout)
            self.log_condor_status(cluster, status)
            return status

        return result

    def check_status(self, debug=False):
        self.log("Checking status...")

        if self.statuses is None:
            statuses = {}

            for cluster in self.connector.cluster_names:
                # Check if the job-file is even present
                job_exists = self.check_file(f"$MTCP_JOB_DIR/job.yaml", cluster, debug=debug)
                if not job_exists:
                    statuses[cluster] = dict(status="missing")
                    continue

                result = self.run_command(f"python3 $MTCP_ROOT/pipeliner/remote/check_status.py", cluster, debug=debug)

                if result.success:
                    status = json.loads(result.stdout)
                    self.log_condor_status(cluster, status)

                    statuses[cluster] = status
                else:
                    statuses[cluster] = dict(status="error", error=result.exit_code, message=result.stderr)

            # Check if the artifact exists locally
            artifacts = {}
            for artifact in self.artifacts:
                artifacts[artifact] = self.check_file(artifact, "local", debug=debug)

            statuses["local"] = dict(status="missing", artifacts=artifacts)

            self.statuses = statuses

        return self.statuses

    def interrupt(self, cluster: str, debug=False):
        self.log("Interrupting...", cluster=cluster)
        result = self.run_command(f"python3 $MTCP_ROOT/pipeliner/remote/interrupt.py", cluster, debug=debug)
        success, stdout, stderr, exit_code = result
        return json.loads(stdout) if success else dict(error=exit_code, message=stderr)

    def delete_artifacts(self, cluster: str, debug=False):
        artifacts_str = ", ".join([cyan(f"\"{artifact}\"") for artifact in self.artifacts])
        self.log(f"Deleting artifacts: {artifacts_str}", cluster=cluster)

        results = {}
        for artifact in self.artifacts:
            result = self.run_command(f"rm -rf {artifact}", cluster, debug=debug)
            results[artifact] = not result.success

        return results

    def check_file(self, file: str, cluster: str, debug=False):
        result = self.run_command(f"ls {file}", cluster, debug=debug, silent=True)
        return result.success

    def delete_artifact(self, artifact, cluster, debug=False):
        self.log(f"Deleting artifact: {cyan(artifact)} from {orange(cluster)}")
        result = self.run_command(f"rm -rf {artifact}", cluster, debug=debug)

        if result.success and self.statuses is not None:
            self.statuses[cluster]["artifacts"][artifact] = False

        return result

    def download_artifact(self, artifact: str, cluster_from: str, cluster_to: str, debug=False):
        self.log(f"Downloading artifact: {cyan(artifact)} from {orange(cluster_from)} to {orange(cluster_to)}")

        cluster_to = self.connector[cluster_to]
        cluster_from = self.connector[cluster_from]

        try:
            cluster_to.get_file(artifact, cluster_from, exports=self.exports, debug=debug)

            if self.statuses is not None:
                self.statuses[cluster_to.name]["artifacts"][artifact] = True

            return dict(success=True)
        except Exception as e:
            return dict(success=False, message=str(e))
