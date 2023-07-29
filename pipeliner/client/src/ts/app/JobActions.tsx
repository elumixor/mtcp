import React, { useContext } from "react";
import { JobStatusesResponse, post } from "server-api";
import { JobDataContext, JobStatusesContext } from "./JobContext";
import { IconButton } from "./IconButton";

export function JobActions({
    cluster,
    setJobStatus,
}: {
    cluster: string;
    setJobStatus: React.Dispatch<React.SetStateAction<JobStatusesResponse | undefined>>;
}) {
    const { name: jobName } = useContext(JobDataContext);
    const jobStatuses = useContext(JobStatusesContext);
    const clusterStatus = jobStatuses?.[cluster] ?? {};

    const hasOut = clusterStatus.out;
    const hasErr = clusterStatus.err;
    const hasLog = clusterStatus.log;

    const show = (what: "log" | "err" | "out") => {
        console.log(clusterStatus[what]);
    };

    let status = jobStatuses?.[cluster]?.status;
    if (status === "error") status = "failed";

    const actionPossible = status !== "missing" && status !== undefined;
    const actionImage = status === undefined ? "trash" : status === "running" ? "stop" : "run";

    const onActionButton = async () => {
        switch (status) {
            case "done":
            case "interrupted":
            case "failed":
            case "not_started": {
                setJobStatus((jobStatus) => {
                    const newJobStatus = { ...jobStatus };
                    newJobStatus[cluster].status = undefined;
                    return newJobStatus;
                });

                const newStatus = await post("run_job", { job: jobName, cluster });

                setJobStatus((jobStatus) => {
                    const newJobStatus = { ...jobStatus };
                    newJobStatus[cluster] = newStatus;
                    return newJobStatus;
                });

                break;
            }
            case "running": {
                setJobStatus((jobStatus) => {
                    const newJobStatus = { ...jobStatus };
                    newJobStatus[cluster].status = undefined;
                    return newJobStatus;
                });

                const newStatus = await post("interrupt_job", { job: jobName, cluster });

                setJobStatus((jobStatus) => {
                    const newJobStatus = { ...jobStatus };
                    newJobStatus[cluster] = newStatus;
                    return newJobStatus;
                });

                break;
            }
        }
    };

    return (
        <span className="job-action">
            <div className={"job-action-cluster" + (actionPossible ? " " : " disabled ") + (status ?? "pending")}>
                {actionPossible ? <IconButton image={actionImage} onClick={onActionButton} /> : <></>}
                <span className="tag-text">{cluster}</span>
            </div>
            <div className="job-action-logs-container">
                <IconButton image="log-out" disabled={!hasOut} onClick={() => show("out")} />
                <IconButton image="log-err" disabled={!hasErr} onClick={() => show("err")} />
                <IconButton image="log-log" disabled={!hasLog} onClick={() => show("log")} />
            </div>
        </span>
    );
}
