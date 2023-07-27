import React, { createContext, useContext, useEffect, useState } from "react";
import { IJobData, IJobStatusResponse, JobStatusesResponse } from "responses";
import { post, requests } from "server-api";
import { mergeStates } from "utils";
import { Artifact } from "./Artifact";
import { StatusIcon } from "./StatusIcon";
import { ReactSVG } from "react-svg";
import { ClustersContext } from "./ClustersContext";

export function Job({ jobData }: { jobData: IJobData }) {
    const jobName = jobData.name;
    const clusters = jobData.clusters;

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

    const onActionButton = (cluster: string, status: string) => {
        switch (status) {
            case "done":
            case "interrupted":
            case "not_started":
            // return runJobRequest({ job: jobName, cluster });
            case "running":
            // return interruptJobRequest({ job: jobName, cluster });
        }
    };

    // Set the interval to check for the status of the job
    // useEffect(() => {
    //     const interval = setInterval(() => {
    //         if (status && Object.values(status).every((status) => status === "running")) {
    //             console.log("checking job status...");
    //             jobStatusRequest({ job: jobName });
    //             console.log(jobStatus);
    //         }
    //     }, 1000);
    //     return () => clearInterval(interval);
    // }, [jobStatus]);

    const [artifactsShown, setArtifactsShown] = useState(true);

    return (
        <div className="job">
            {/* {status === "missing" || status === "..." ? (
                <></>
            ) : (
                <ReactSVG
                    className={"button job-refresh" + (blocked ? " disabled" : "")}
                    onClick={() => jobStatusRequest({ job: jobName })}
                    src="icons/refresh.svg"
                />
            )} */}
            <div className="job-main">
                <div className="job-info text-left">
                    <h4 className="job-title">{jobName}</h4>
                    <div className="job-info">{jobData.description}</div>
                </div>
                <div className="job-actions text-right">
                    {/* <StatusIcon status={status} /> */}
                    {clusters.map((cluster) => {
                        const status = jobStatus?.[cluster]?.status ?? "...";
                        return status !== "missing" ? (
                            <button
                                key={cluster}
                                disabled={status !== "running"}
                                onClick={() => onActionButton(cluster, status)}
                            >
                                {status === "done" || status === "interrupted"
                                    ? "Restart"
                                    : status === "running"
                                    ? "Interrupt"
                                    : status === "not_started"
                                    ? "Start"
                                    : "..."}
                            </button>
                        ) : (
                            <></>
                        );
                    })}
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
