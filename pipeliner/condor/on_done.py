#!/usr/bin/env python3

import os
import sys
import json

job_folder = sys.argv[1]

# Read the status file and parse the json
status_file = os.path.join(job_folder, "status.json")
with open(status_file, "r") as f:
    status = json.load(f)

status["status"] = "done"

# Write the status file
with open(status_file, "w") as f:
    f.write(json.dumps(status))

# Print the status
print(json.dumps(status))