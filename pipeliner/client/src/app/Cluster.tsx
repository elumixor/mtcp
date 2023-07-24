import React, { useEffect } from "react";
import { IConnectResponse, IJobData } from "responses";
import { requests } from "server-api";
import { Job } from "./Job";

export function Cluster(props: { name: string; jobs: IJobData[] }) {
    const isLocal = props.name === "[local]";

    const [blocked, states] = requests(
        { endpoint: "connect", params: { cluster: props.name } },
        { endpoint: "git_sync", params: { cluster: props.name } },
    );
    const [[connectResponse, connect], [gitSyncResponse, gitSync]] = states;

    const connected = (connectResponse as IConnectResponse)?.connected ?? false;
    const { status, error } = (gitSyncResponse ?? {}) as Partial<{ status: string; error: string }>;

    if (!isLocal) useEffect(() => void connect(), [props.name]);

    return (
        <div className="cluster">
            <h2>{props.name}</h2>
            {isLocal ? (
                <></>
            ) : (
                <>
                    <button
                        disabled={blocked || connected}
                        className={connected ? "good" : ""}
                        onClick={() => connect()}
                    >
                        {blocked ? "..." : connected ? "Connected" : "Connect"}
                    </button>
                    <button disabled={blocked || !connected} onClick={() => gitSync()}>
                        Sync
                    </button>
                </>
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
