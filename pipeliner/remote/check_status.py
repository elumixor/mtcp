import os
import json

from run_command import run_command


def get_artifacts():
    # Also check job artifacts if they exist or no
    job_config = os.path.join(job_folder, "job.yaml")
    with open(job_config, "r") as f:
        # Find the line starting with "artifacts:"
        for line in f:
            if line.startswith("artifacts:"):
                break

        if "[" in line:
            line = line[line.index("[") + 1:line.index("]")]
            return [a.strip() for a in line.split(",")]

        artifacts = []
        for line in f:
            line = line.strip()
            # Each item should start with "-"
            if line.startswith("-"):
                artifacts.append(line[1:].strip())
            else:
                break

        return artifacts


job_name = os.environ["MTCP_JOB"]
job_folder = os.environ["MTCP_JOB_DIR"]

# Get the job status file
status_file = os.path.join(job_folder, "status.json")

# If it doesn't exist, the job hasn't started yet
if not os.path.exists(status_file):
    # Check if artifacts exist on disk
    status = dict(
        status="not_started",
        artifacts={a: os.path.exists(os.path.expandvars(a)) for a in get_artifacts()}
    )

    print(json.dumps(status))
    exit(0)

# If it does exist, read it and print its contents
with open(status_file, "r") as f:
    # Read the file and parse the json
    status = json.load(f)

# If the status contains "condor:id", then it is a condor job and we should include more information
# print(status)
if "condor" in status:
    condor = status["condor"]
    condor_id = condor["id"]

    # If the state is still "running" then we should check the condor's status, otherwise set as done
    if status["status"] == "running":
        output = run_command(f"condor_q {condor_id} -json")
        if output.strip() != "":
            output = json.loads(output)[0]

            # Add the condor status to the status
            job_status = output["JobStatus"]
            status["condor"]["status"] = "hold" if job_status == 5 else \
                "idle" if job_status == 1 else \
                "running" if job_status == 2 else \
                "done" if job_status == 4 else \
                "error"

            status["status"] = "error" if job_status == 5 else \
                "running" if job_status == 1 else \
                status["condor"]["status"]
        else:
            status["condor"]["status"] = "done"
            status["status"] = "done"
    else:
        status["condor"]["status"] = status["status"]

# Add the log
log_path = os.path.join(job_folder, "logs/log")
if os.path.exists(log_path):
    status["log"] = log_path

# Add out and err from the files
out_path = os.path.join(job_folder, "logs/out")
if os.path.exists(out_path):
    status["out"] = out_path

err_path = os.path.join(job_folder, "logs/err")
if os.path.exists(err_path):
    status["err"] = err_path

# Check if artifacts exist on disk
status["artifacts"] = {a: os.path.exists(os.path.expandvars(a)) for a in get_artifacts()}

# Print the status
print(json.dumps(status))
