import autorootcwd  # Do not delete - adds the root of the project to the path

from time import sleep

from pipeliner.connector import Connector
from pipeliner.job_runner import JobRunner
from pipeliner.server import Server


if __name__ == "__main__":
    # with Connector() as connector, JobRunner() as runner, Server() as server:
    with Connector() as connector, JobRunner() as runner:
        print(runner["long-touch"].run("cern"))
        # try:
        #     while True:
        #         sleep(1)
        # except KeyboardInterrupt:
        #     pass

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
