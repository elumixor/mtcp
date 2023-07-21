from pipeliner.connector import Connector
from pipeliner.utils import inject, magenta, orange, red, green, cyan, DotDict


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

    def get_exports(self, cluster: str, root: str):
        return f"export P_CLUSTER={cluster}" + \
            f" && export P_JOB_NAME={self.name}" + \
            f" && export P_ROOT={root}/mtcp" + \
            f" && export P_ARTIFACTS=$P_ROOT/artifacts" + \
            f" && export P_JOBS=$P_ROOT/jobs"

    def run(self, cluster: str, clean=True):
        if clean:
            self.delete_artifacts(cluster)

        with self.connector[cluster] as connection:
            self.log("Running the job", cluster=cluster)

            root = connection.root

            if cluster == "cern" and self.condor:
                self.log("Submitting to condor", cluster=cluster)
                cmd = f"cd {root} && {self.get_exports(cluster, root)} && python ./pipeliner/condor.py {self.command}"
            else:
                cmd = f"cd {root} && {self.get_exports(cluster, root)} && {self.command}"

            result = connection.run_command(cmd)

            if not result.success:
                self.log("Failed. See output below:", cluster=cluster, error=True)
                print(result.stderr.strip())

            if result.stdout:
                self.log(result.stdout, cluster=cluster)

            return result

    def check_artifacts(self, cluster: str):
        results = {}

        with self.connector[cluster] as connection:
            self.log("Checking artifacts", cluster=cluster)

            root = connection.root

            for artifact in self.artifacts:
                cmd = f"cd {root} && {self.get_exports(cluster, root)} && ls {artifact}"
                result = connection.run_command(cmd)

                results[artifact] = result.success

                self.log("Artifact " + cyan(f"\"{artifact}\" ") +
                         (green("exists") if result.success else red("does not exist")),
                         cluster=cluster)

        return results

    def delete_artifacts(self, cluster: str):
        results = {}
        with self.connector[cluster] as connection:

            artifacts = self.artifacts
            root = connection.root

            artifacts_str = ", ".join([cyan(f"\"{artifact}\"") for artifact in artifacts])
            self.log(f"Deleting artifacts: {artifacts_str}", cluster=cluster)

            for artifact in artifacts:
                cmd = f"cd {root} && {self.get_exports(cluster, root)} && rm -rf {artifact}"

                result = connection.run_command(cmd)

                if not result.success:
                    self.log(result.stderr, cluster=cluster, error=True)
                else:
                    self.log(green("Deleted") + " " + cyan(f"\"{artifact}\""), cluster=cluster)

                results[artifact] = result.success

        return results
