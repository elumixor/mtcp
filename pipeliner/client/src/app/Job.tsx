import React, { useEffect, useState } from "react";
import { IJobData, IJobStatusResponse } from "responses";
import { requests } from "server-api";
import { mergeStates } from "utils";
import { Artifact } from "./Artifact";
import { StatusIcon } from "./StatusIcon";
import { ReactSVG } from "react-svg";

export function Job(props: { cluster: string; jobData: IJobData }) {
    const [blocked, states] = requests(
        { endpoint: "job_status", params: { cluster: props.cluster, job: props.jobData.name } },
        { endpoint: "run_job", params: { cluster: props.cluster, job: props.jobData.name } },
        { endpoint: "interrupt_job", params: { cluster: props.cluster, job: props.jobData.name } },
        { endpoint: "delete_artifacts", params: { cluster: props.cluster, job: props.jobData.name } },
    );

    const [
        [jobStatusResponse, jobStatusRequest],
        [runJobResponse, runJobRequest],
        [interruptJobResponse, interruptJobRequest],
        [deleteArtifactsResponse, deleteArtifactsRequest],
    ] = states;

    const [jobStatus] = mergeStates(jobStatusResponse, runJobResponse, interruptJobResponse);

    useEffect(() => void jobStatusRequest(), [props.cluster, props.jobData.name, deleteArtifactsResponse]);
    // useEffect(() => {
    //     if (!jobStatus.success) ...
    // }, [jobStatus]);

    const { artifacts = {} } = jobStatusResponse ?? {};
    const numArtifacts = Object.keys(artifacts).length;

    const status = jobStatus?.status ?? "...";
    const blockedDelete = blocked || status === "running";

    const condorId = jobStatus?.condor?.id;

    const onActionButton = () => {
        switch (status) {
            case "done":
            case "interrupted":
            case "not_started":
                return runJobRequest();
            case "running":
                return interruptJobRequest();
        }
    };

    // Set the interval to check for the status of the job
    useEffect(() => {
        const interval = setInterval(() => {
            if (status === "running") {
                console.log("checking job status...");
                jobStatusRequest();
                console.log(jobStatus);
            }
        }, 1000);
        return () => clearInterval(interval);
    }, [jobStatus]);

    const [artifactsShown, setArtifactsShown] = useState(true);

    return (
        <div className="job">
            <ReactSVG
                className={"button job-refresh" + (blocked ? " disabled" : "")}
                onClick={jobStatusRequest}
                src="icons/refresh.svg"
            />
            <div className="flex">
                <div className="job-info text-left">
                    <h4 className="job-title">{props.jobData.name}</h4>
                    <div className="job-info">{props.jobData.description}</div>
                </div>
                <div className="job-actions text-right">
                    <StatusIcon status={status} />
                    {status !== "..." ? (
                        <button disabled={blocked && status !== "running"} onClick={onActionButton}>
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
                    )}
                    {numArtifacts > 0 ? (
                        <button disabled={blockedDelete} onClick={deleteArtifactsRequest}>
                            Delete Artifacts
                        </button>
                    ) : (
                        <></>
                    )}
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
                    Object.entries(artifacts).map(([artifact, exists]) => (
                        <Artifact key={artifact} artifact={artifact} exists={exists} />
                    ))
                ) : (
                    <></>
                )}
            </div>
        </div>
    );
}
