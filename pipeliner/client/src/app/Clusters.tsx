import React, { useEffect, useState } from "react";
import { IArtifactStatus, IJobData } from "responses";
import { post } from "utils";

export function Clusters(props: { clusters: string[]; jobs: IJobData[] }) {
    return (
        <div id="clusters">
            <Cluster name="[local]" key="[local]" jobs={[]} />
            {props.clusters.map((cluster) => (
                <Cluster
                    name={cluster}
                    jobs={props.jobs.filter((job) => job.clusters.includes(cluster))}
                    key={cluster}
                />
            ))}
        </div>
    );
}

export function Cluster(props: { name: string; jobs: IJobData[] }) {
    const isLocal = props.name === "[local]";
    const [connected, setConnected] = useState(isLocal ? true : null);

    if (!isLocal) {
        useEffect(() => {
            post("/check_connection", { cluster: props.name }).then(({ connected }) => setConnected(connected));
        }, [props.name]);
    }

    return (
        <div className="cluster">
            <h2>{props.name}</h2>
            {isLocal ? (
                <></>
            ) : (
                <button
                    disabled={connected || connected === null}
                    className={connected ? "good" : ""}
                    onClick={() => {
                        setConnected(null);
                        post("/connect", { cluster: props.name }).then(({ connected }) => setConnected(connected));
                    }}
                >
                    {connected === null ? "..." : connected ? "Connected" : "Connect"}
                </button>
            )}
            {!connected ? (
                <></>
            ) : (
                <div className="jobs">
                    {props.jobs.map((job) => (
                        <Job key={job.name} cluster={props.name} jobData={job} />
                    ))}
                </div>
            )}
        </div>
    );
}

export function Job(props: { cluster: string; jobData: IJobData }) {
    const [running, setRunning] = useState(true);
    const [artifacts, setArtifacts] = useState<IArtifactStatus>({});

    const checkArtifacts = async () => {
        const artifacts = await post("/check_artifacts", { cluster: props.cluster, job: props.jobData.name });
        setArtifacts(artifacts);
    };

    useEffect(() => void checkArtifacts().then(() => setRunning(false)), [props.cluster, props.jobData.name]);

    const handleRunJob = async () => {
        setRunning(true);
        await post("/run_job", { cluster: props.cluster, job: props.jobData.name });
        await checkArtifacts();
        setRunning(false);
    };

    const handleDeleteArtifacts = async () => {
        setRunning(true);
        await post("/delete_artifacts", { cluster: props.cluster, job: props.jobData.name });
        await checkArtifacts();
        setRunning(false);
    };

    return (
        <div className="job">
            <h4>{props.jobData.name}</h4>
            <div className="job-info">{props.jobData.description}</div>
            <div className="job-actions">
                <button disabled={running} onClick={handleRunJob}>
                    {running ? "Running..." : "Run Job"}
                </button>
                <button disabled={running} onClick={handleDeleteArtifacts}>
                    Delete Artifacts
                </button>
            </div>
            {Object.entries(artifacts).length > 0 ? <hr /> : <></>}
            <div className="job-artifacts">
                {Object.entries(artifacts).map(([artifact, exists]) => (
                    <Artifact key={artifact} artifact={artifact} exists={exists} />
                ))}
            </div>
        </div>
    );
}

export function Artifact(props: { artifact: string; exists: boolean }) {
    return (
        <div className="artifacts">
            <span className={"artifact-name"}>{props.artifact}</span>
            <span className={"artifact-exists " + (props.exists ? "good" : "bad")}>
                {props.exists ? "Exists" : "Missing"}
            </span>
        </div>
    );
}
