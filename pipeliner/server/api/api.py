from pipeliner.utils import inject
from pipeliner.connector import Connector
from pipeliner.job_runner import JobRunner

from .decorators import get, post


class Api:
    def __init__(self):
        self.connector = inject(Connector)
        self.job_runner = inject(JobRunner)

    @post("/connect")
    def connect(self, cluster):
        return {"connected": self.connector[cluster].open()}

    @post("/clusters")
    def get_clusters(self):
        return self.connector.cluster_names

    @post("/jobs")
    def get_jobs(self):
        return [job.json for job in self.job_runner.jobs.values()]

    @post("/git_sync")
    def git_sync(self, cluster=None, debug=False):
        return self.connector.sync(cluster=cluster, debug=debug)

    @post("/job_status")
    def job_status(self, job, debug=False):
        return self.job_runner[job].check_status(debug=debug)

    @post("/interrupt_job")
    def interrupt_job(self, job, cluster, debug=False):
        return self.job_runner[job].interrupt(cluster, debug=debug)

    @post("/delete_artifact")
    def delete_artifact(self, artifact, job, cluster, debug=False):
        return self.job_runner[job].delete_artifact(artifact, cluster, debug=debug)

    @post("/download_artifact")
    def download_artifact(self, artifact, job, cluster_from, cluster_to, debug=False):
        return self.job_runner[job].download_artifact(artifact, cluster_from, cluster_to, debug=debug)

    @post("/run_job")
    def run_job(self, job, cluster, debug=False):
        return self.job_runner[job].run(cluster, debug=debug)

    # @post("/get_commit")
    # def get_git_status(self):
    #     output = subprocess.check_output(["git", "rev-parse", "HEAD"])
    #     return output.decode("utf-8")

    # @post("/git_pull")
    # def git_pull(self):
    #     output = subprocess.check_output(["git", "pull"])
    #     return output.decode("utf-8")

    # @post("/git_push")
    # def git_push(self):
    #     output = subprocess.check_output(["git", "push"])
    #     return output.decode("utf-8")
