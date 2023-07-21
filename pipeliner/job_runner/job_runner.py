import os

from pipeliner.utils import read_yaml, auto_provided
from .job import Job


@auto_provided
class JobRunner:
    def __init__(self):
        # Scan the jobs directory for jobs
        self.job_directory = os.path.join(os.path.abspath(os.curdir), "jobs")
        self.jobs: dict[str, Job] = {}
        for root, _, files in os.walk(self.job_directory):
            if "job.yaml" in files:
                # Take the name as everything after the self.job_directory
                job_name = root[len(self.job_directory) + 1:]
                job_config = read_yaml(os.path.join(root, "job.yaml"))
                self.jobs[job_name] = Job(job_name, job_config)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass

    def __getitem__(self, key):
        return self.jobs[key]
