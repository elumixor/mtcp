import React, { useContext, useEffect, useState } from "react";
import { IJobData, JobStatusesResponse } from "server-api";
import { post } from "server-api";
import { Artifact } from "./Artifact";
import { ClustersContext } from "./ClustersContext";
import { JobActions } from "./JobActions";
import { JobDataContext, JobStatusesContext } from "./JobContext";

export function Job({ jobData }: { jobData: IJobData }) {
    const jobName = jobData.name;

    const allClusters = useContext(ClustersContext);

    const [jobStatus = {}, setJobStatus] = useState<JobStatusesResponse>();

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

    // Log the job status
    useEffect(() => {
        if (Object.keys(jobStatus).length <= 0) return;

        console.log(`%c${jobName}`, "font-weight: bold; color: #f4a;");
        for (const cluster of jobData.clusters) {
            const status = jobStatus[cluster]?.status ?? "pending";
            console.log(`%c${cluster}:`, "color: #8d9;", status);
        }
    }, [jobStatus]);

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
    // useEffect(() => {
    //     const interval = setTimeout(async () => {
    //         if (jobStatus && Object.values(jobStatus).some((cluster) => cluster.status === "running")) {
    //             const newStatus = await post("job_status", { job: jobName });
    //             setJobStatus(newStatus);
    //         }
    //     }, 1000);
    //     return () => clearInterval(interval);
    // }, [jobStatus]);

    const [artifactsShown, setArtifactsShown] = useState(true);

    return (
        <JobDataContext.Provider value={jobData}>
            <JobStatusesContext.Provider value={jobStatus}>
                <div className="job">
                    <div className="job-main">
                        <div className="job-info text-left">
                            <h4 className="job-title">{jobName}</h4>
                            <div className="job-info">{jobData.description}</div>
                        </div>
                        <div className="job-actions text-right">
                            {jobData.clusters.map((cluster) => (
                                <JobActions key={cluster} cluster={cluster} setJobStatus={setJobStatus} />
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
                                <Artifact key={name} artifact={name} exists={exists} setJobStatus={setJobStatus} />
                            ))
                        ) : (
                            <></>
                        )}
                    </div>
                </div>
            </JobStatusesContext.Provider>
        </JobDataContext.Provider>
    );
}
