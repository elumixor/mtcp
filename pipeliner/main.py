import autorootcwd  # Do not delete - adds the root of the project to the path

from time import sleep

from pipeliner.connector import Connector
from pipeliner.job_runner import JobRunner
from pipeliner.server import Server
from pipeliner.utils import inject


# def start_job():
#     runner = inject(JobRunner)

#     status = runner["long-touch"].check_status("cern")

#     if status["status"] == "running":
#         choice = ""
#         while choice not in ["y", "n"]:
#             choice = input("Already in progress. Restart? [y/n] ")

#         if choice == "y":
#             runner["long-touch"].interrupt("cern")
#             runner["long-touch"].run("cern")
#             return
#         else:
#             print("Continuing...")
#             return

#     elif status["status"] == "done":
#         choice = ""
#         while choice not in ["y", "n"]:
#             choice = input("Already done. Restart? [y/n] ")

#         if choice == "n":
#             return

#         runner["long-touch"].run("cern")


# def monitor_job():
#     try:
#         while True:
#             status = runner["long-touch"].check_status("cern")
#             if status["status"] == "done":
#                 print("Done")
#                 break
#             sleep(1)
#     except KeyboardInterrupt:
#         pass


if __name__ == "__main__":
    with Connector() as connector, JobRunner() as runner, Server() as server:
        # start_job()
        # monitor_job()
        try:
            while True:
                sleep(1)
        except KeyboardInterrupt:
            pass

    # runner.jobs["initialize"].run(connector)
    # runner.jobs["touch"].run(connector)
    # print(runner.jobs["initialize"].check_artifacts(connector))
    # print(runner.jobs["initialize"].clean_artifacts(connector))

    # print("connector created")

    # with connector.cern as cern:
    #     print("cern connected")
    #     print(cern.run_command("ls")[0])

    # with connector.labmda as lambda:
    #     print("lambda connected")
    #     print(lambda.run_command("ls")[0])

    # print("connector closed")
