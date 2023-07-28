import React, { createContext, useContext, useEffect, useState } from "react";
import { IJobData, IJobStatusResponse, JobRunState, JobStatusesResponse } from "responses";
import { post, requests } from "server-api";
import { mergeStates } from "utils";
import { Artifact } from "./Artifact";
import { StatusIcon } from "./StatusIcon";
import { ReactSVG } from "react-svg";
import { ClustersContext } from "./ClustersContext";

export function Job({ jobData }: { jobData: IJobData }) {
    const jobName = jobData.name;

    const allClusters = useContext(ClustersContext);

    const [jobStatus, setJobStatus] = useState<JobStatusesResponse>();

    // const [blocked, states] = requests(
    //     { endpoint: "job_status" },
    //     { endpoint: "run_job" },
    //     { endpoint: "interrupt_job" },
    // );

    // const [
    //     [jobStatusResponse, jobStatusRequest],
    //     [runJobResponse, runJobRequest],
    //     [interruptJobResponse, interruptJobRequest],
    // ] = states;

    // const [jobStatus] = mergeStates(runJobResponse, interruptJobResponse);

    const getJobStatus = async () => {
        const status = await post("job_status", { job: jobName });
        setJobStatus(status);
    };

    useEffect(() => void getJobStatus(), [jobName]);

    const artifacts = jobData.artifacts.map(
        (artifact) =>
            [
                artifact,
                Object.fromEntries(
                    allClusters.map((cluster) => [cluster, jobStatus?.[cluster]?.artifacts?.[artifact] ?? null]),
                ) as Record<string, boolean | null>,
            ] as const,
    );

    const numArtifacts = jobData.artifacts.length;

    // const status = jobStatus?.status;
    // const blockedDelete = blocked || status === "running";

    // const condorId = jobStatus?.condor?.id;

    // Set the interval to check for the status of the job
    useEffect(() => {
        const interval = setTimeout(async () => {
            if (jobStatus && Object.values(jobStatus).some((cluster) => cluster.status === "running")) {
                const newStatus = await post("job_status", { job: jobName });
                setJobStatus(newStatus);
            }
        }, 1000);
        return () => clearInterval(interval);
    }, [jobStatus]);

    const [artifactsShown, setArtifactsShown] = useState(true);

    return (
        <div className="job">
            <div className="job-main">
                <div className="job-info text-left">
                    <h4 className="job-title">{jobName}</h4>
                    <div className="job-info">{jobData.description}</div>
                </div>
                <div className="job-actions text-right">
                    {allClusters.map((cluster) => (
                        <RunButton
                            key={cluster}
                            cluster={cluster}
                            jobData={jobData}
                            jobStatus={jobStatus}
                            setJobStatus={setJobStatus}
                        />
                    ))}
                </div>
            </div>
            {numArtifacts > 0 ? (
                <>
                    <hr />
                    <div className="button" onClick={() => setArtifactsShown(!artifactsShown)}>
                        {artifactsShown ? "Hide" : "Show"} Artifacts ({numArtifacts})
                    </div>
                </>
            ) : (
                <></>
            )}
            <div className="job-artifacts-section">
                {artifactsShown ? (
                    artifacts.map(([name, exists]) => (
                        <Artifact
                            key={name}
                            job={jobName}
                            artifact={name}
                            exists={exists}
                            setJobStatus={setJobStatus}
                        />
                    ))
                ) : (
                    <></>
                )}
            </div>
        </div>
    );
}

export function RunButton({
    cluster,
    jobData: { name: jobName },
    jobStatus,
    setJobStatus,
}: {
    cluster: string;
    jobData: IJobData;
    jobStatus: JobStatusesResponse | undefined;
    setJobStatus: React.Dispatch<React.SetStateAction<JobStatusesResponse | undefined>>;
}) {
    const status = jobStatus?.[cluster]?.status;
    const actionPossible = status !== "missing" && status !== undefined;
    const actionImage = "run";

    const onActionButton = async () => {
        switch (status) {
            case "done":
            case "interrupted":
            case "failed":
            case "not_started": {
                setJobStatus((jobStatus) => {
                    const newJobStatus = { ...jobStatus };
                    newJobStatus[cluster].status = "pending";
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
                    newJobStatus[cluster].status = "pending";
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
        <span
            className={"job-action" + (actionPossible ? "" : " disabled")}
            // className={"artifact-exists " + (clusterExists === null ? "pending" : clusterExists ? "good" : "neutral")}
        >
            {actionPossible ? (
                <span className={"artifact-image" + (actionPossible ? " button" : "")} onClick={onActionButton}>
                    <ReactSVG src={`assets/${actionImage}.svg`} />
                </span>
            ) : (
                <></>
            )}
            {cluster}
        </span>
    );
}
