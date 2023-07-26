import sys
import os
import json
import argparse

from run_command import run_command

# Get the submission command
parser = argparse.ArgumentParser()
parser.add_argument("command", help="The command to submit")
parser.add_argument("--max-runtime", help="The time (in minute) to run the job for. Default is 30m", default="30")

args = parser.parse_args()
command = args.command
max_runtime = args.max_runtime * 60

# Get the job dir from the env
mtcp_folder = os.environ["MTCP_ROOT"]
job_name = os.environ["MTCP_JOB"]
job_folder = os.environ["MTCP_JOB_DIR"]
artifacts_folder = os.environ["MTCP_JOB_ARTIFACTS_DIR"]
logs_folder = os.environ["MTCP_JOB_LOGS_DIR"]

# Create the job artifacts folder if it doesn't exist
if not os.path.exists(artifacts_folder):
    os.makedirs(artifacts_folder)

# Unfortunately, we cannot create log space on eos
# So we will create an $HOME/.condor/logs folder and make a symlink to it
condor_home_folder = os.path.join(os.environ["HOME"], ".condor")
job_home_folder = os.path.join(condor_home_folder, job_name)
logs_home_folder = os.path.join(job_home_folder, "logs")
if not os.path.exists(logs_home_folder):
    os.makedirs(logs_home_folder)

# Remove the old logs
os.system(f"rm -rf {logs_home_folder}/*")

# Copy the `on_done.py` script to the condor home folder
on_done_file = os.path.join(mtcp_folder, "pipeliner/condor/on_done.py")

# Create the command home folder if it doesn't exist
command_home_folder = os.path.join(job_home_folder, "commands")
if not os.path.exists(command_home_folder):
    os.makedirs(command_home_folder)

# Copy the command to the command home folder
file_name = os.path.basename(command)
command_home_file = os.path.join(command_home_folder, file_name)
os.system(f"cp {command} {command_home_file}")

# Remove the eos job folder if it exists
if os.path.exists(logs_folder):
    os.system(f"rm -rf {logs_folder}")

# Create the symlink
os.system(f"ln -s {logs_home_folder} {logs_folder}")

# Get arguments
arguments = "" if len(sys.argv) == 2 else " ".join(sys.argv[2:])

stdin_command = f"""
executable            = {command_home_file}
arguments             = {arguments}
output                = {logs_home_folder}/out
error                 = {logs_home_folder}/err
log                   = {logs_home_folder}/log
initialdir            = {job_home_folder}
getenv                = MTCP*

+JobFlavour           = "espresso"
+MaxRuntime           = {max_runtime}

+PostCmd              = "/usr/bin/python3"
+PostArguments        = "{on_done_file} {job_folder}"

queue
"""

# Write the file to the job folder
submit_file = os.path.join(job_folder, "condor_submit.sub")
with open(submit_file, "w") as f:
    f.write(stdin_command)

# Check if the pre-condor.sh file exists. If it does, run it
pre_condor_file = os.path.join(job_folder, "pre-condor.sh")
if os.path.exists(pre_condor_file):
    run_command(f"bash {pre_condor_file}")

# Call the condor_submit with the file
stdout = run_command(f"condor_submit -terse {submit_file}")

# Get the id
id = stdout.strip()
id = id.split(".")[0]

# Set the status to "started"
status = json.dumps(dict(status="running", condor=dict(id=id, status="idle")))

# Write the status to the status file
status_file = os.path.join(job_folder, "status.json")
with open(status_file, "w") as f:
    f.write(status)

# Print the status
print(status)