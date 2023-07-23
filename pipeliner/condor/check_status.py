import os
import json

from run_command import run_command

job_name = os.environ["MTCP_JOB"]
job_folder = os.environ["MTCP_JOB_DIR"]

# Get the job status file
status_file = os.path.join(job_folder, "status.json")

# If it doesn't exist, the job hasn't started yet
if not os.path.exists(status_file):
    print(json.dumps({"status": "not_started"}))
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

    # Add the log
    log_path = os.path.join(job_folder, "logs/log")
    if os.path.exists(log_path):
        with open(log_path, "r") as f:
            status["condor"]["log"] = f.read().strip()

    # If the state is still "running" then we should check the condor's status, otherwise set as done
    output = run_command(f"condor_q {condor_id} -json")
    if status["status"] == "running" and output.strip() != "":
        output = json.loads(output)[0]

        # Add the condor status to the status
        job_status = output["JobStatus"]
        status["condor"]["status"] = "hold" if job_status == 5 else \
                                     "idle" if job_status == 1 else \
                                     "running" if job_status == 2 else \
                                     "done" if job_status == 4 else \
                                     "error"
    else:
        status["condor"]["status"] = status["status"]

        # Add out and err from the files
        out_file = os.path.join(job_folder, "logs/out")
        if os.path.exists(out_file):
            with open(out_file, "r") as f:
                status["out"] = f.read().strip()

        err_file = os.path.join(job_folder, "logs/err")
        if os.path.exists(err_file):
            with open(err_file, "r") as f:
                status["err"] = f.read().strip()

# Print the status
print(json.dumps(status))

