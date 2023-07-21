import subprocess

from pipeliner.utils import inject
from pipeliner.connector import Connector
from pipeliner.job_runner import JobRunner

from .decorators import get, post


class Api:
    def __init__(self):
        self.connector = inject(Connector)
        self.job_runner = inject(JobRunner)

    @post("/clusters")
    def get_clusters(self):
        return list(self.connector.connections.keys())

    @post("/jobs")
    def get_jobs(self):
        return [job.json() for job in self.job_runner.jobs.values()]

    @post("/check_connection")
    def check_connection(self, cluster):
        return {"connected": self.connector[cluster].is_connected}

    @post("/connect")
    def connect(self, cluster):
        return {"connected": self.connector[cluster].open()}

    @post("/check_artifacts")
    def check_artifacts(self, job, cluster):
        return self.job_runner[job].check_artifacts(cluster)

    @post("/delete_artifacts")
    def delete_artifacts(self, job, cluster):
        return self.job_runner[job].delete_artifacts(cluster)

    @post("/run_job")
    def run_job(self, job, cluster):
        return self.job_runner[job].run(cluster)

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
