import React, { useEffect } from "react";
import { IConnectResponse, IJobData } from "responses";
import { requests } from "server-api";
import { Job } from "./Job";
import { ReactSVG } from "react-svg";

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
            <div className="cluster-header">
                <h2 className="cluster-title">
                    {props.name}
                    <ReactSVG className="button job-refresh" onClick={gitSync} src="icons/refresh.svg" />
                </h2>
            </div>
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
                </>
            )}
            {!connected ? (
                <></>
            ) : (
                <div className={"jobs" + (blocked ? " disabled alpha-50" : "")}>
                    {props.jobs.map((job) => (
                        <Job key={job.name} cluster={props.name} jobData={job} />
                    ))}
                </div>
            )}
        </div>
    );
}
