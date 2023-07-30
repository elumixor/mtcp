import os
import sys
import json

from run_command import run_command

# Get arguments
mtcp_folder = os.environ["MTCP_ROOT"]
job_name = os.environ["MTCP_JOB"]
job_folder = os.environ["MTCP_JOB_DIR"]

# Get the job status file
status_file = os.path.join(job_folder, "status.json")

# Read it and json-parse it
with open(status_file, "r") as f:
    status = json.load(f)

if "condor" in status:
    if "id" not in status["condor"]:
        raise Exception("Condor job has no id!")

    # condor_rm `cluster_id`
    cluster_id = status["condor"]["id"]
    run_command(f"condor_rm {cluster_id}")
elif "tmux" in status:
    if "session" not in status["tmux"]:
        raise Exception("Tmux job has no id!")

    # tmux kill-session -t `session_name`
    session_name = status["tmux"]["session"]
    run_command(f"tmux kill-session -t {session_name}")

# Change the status field to "interrupted"
status["status"] = "interrupted"

# Write the status to the status file
with open(status_file, "w") as f:
    f.write(json.dumps(status))

# Print the status
print(json.dumps(status))
