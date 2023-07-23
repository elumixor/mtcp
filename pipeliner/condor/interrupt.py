import os
import sys
import json
import subprocess

# Get arguments
mtcp_folder = os.environ["MTCP_ROOT"]
job_name = os.environ["MTCP_JOB"]
job_folder = os.environ["MTCP_JOB_DIR"]

# Get the job status file
status_file = os.path.join(job_folder, "status.json")

# Read it and json-parse it
with open(status_file, "r") as f:
    status = json.load(f)

# condor_rm `cluster_id`
cluster_id = status["condor"]["id"]

# Call the condor_submit with the file
process = subprocess.Popen(
    f"condor_rm {cluster_id}",
    shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
)

stdout = process.stdout.read().decode("utf-8")
stderr = process.stderr.read().decode("utf-8")
exit_code = process.poll()

if exit_code != 0:
    print("condor_rm failed!", file=sys.stderr)
    print(stderr, file=sys.stderr)
    exit(1)

# Change the status field to "interrupted"
status["status"] = "interrupted"

# Write the status to the status file
with open(status_file, "w") as f:
    f.write(json.dumps(status))

# Print the status
print(json.dumps(status))