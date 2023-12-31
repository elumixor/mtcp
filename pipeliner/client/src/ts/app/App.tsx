import React, { useEffect, useState } from "react";
import { post } from "server-api";
import { IConnectResponse, IJobData } from "server-api";
import { requests } from "server-api";
import { Job } from "./Job";
import { ReactSVG } from "react-svg";
import { ClustersContext } from "./ClustersContext";
import { all } from "utils";

export function App() {
    const [clusters, setClusters] = useState<string[]>([]);
    const [jobs, setJobs] = useState<IJobData[]>([]);
    const [blocked, setBlocked] = useState(false);

    const update = async () => {
        const [clusters, jobs] = await all(post("clusters"), post("jobs"));
        setClusters(clusters);
        setJobs(jobs);
    };

    const sync = async () => {
        setJobs([]);

        const status = await post("git_sync", { debug: true });
        console.log(status);

        await update();
    };

    useEffect(() => void update(), []);

    return (
        // {!connected ? (
        <>
            <h1 className="title">MTCP</h1>
            <button onClick={sync} className="sync-button">
                Sync
            </button>
            <ClustersContext.Provider value={clusters}>
                <div className={"jobs" + (blocked ? " disabled alpha-50" : "")}>
                    {jobs.map((job) => (
                        <Job key={job.name} jobData={job} />
                    ))}
                </div>
            </ClustersContext.Provider>
        </>
        // )}
    );

    // return (
    //     <>
    //         <div id="clusters">
    //             {clusters.map((cluster) => (
    //                 <Cluster name={cluster} jobs={jobs.filter((job) => job.clusters.includes(cluster))} key={cluster} />
    //             ))}
    //         </div>
    //     </>
    // );
}

// export function Cluster(props: { name: string; jobs: IJobData[] }) {
//     const isLocal = props.name === "[local]";

//     const [blocked, states] = requests(
//         { endpoint: "connect", params: { cluster: props.name } },
//         { endpoint: "git_sync", params: { cluster: props.name } },
//     );
//     const [[connectResponse, connect], [gitSyncResponse, gitSync]] = states;

//     const connected = (connectResponse as IConnectResponse)?.connected ?? false;
//     const { status, error } = (gitSyncResponse ?? {}) as Partial<{ status: string; error: string }>;

//     if (!isLocal) useEffect(() => void connect(), [props.name]);

//     return (
//         <div className="cluster">
//             <div className="cluster-header">
//                 <h2 className="cluster-title">
//                     {props.name}
//                     <ReactSVG className="button job-refresh" onClick={gitSync} src="icons/refresh.svg" />
//                 </h2>
//             </div>
//             {isLocal ? (
//                 <></>
//             ) : (
//                 <>
//                     <button
//                         disabled={blocked || connected}
//                         className={connected ? "good" : ""}
//                         onClick={() => connect()}
//                     >
//                         {blocked ? "..." : connected ? "Connected" : "Connect"}
//                     </button>
//                 </>
//             )}
//             {!connected ? (
//                 <></>
//             ) : (
//                 <div className={"jobs" + (blocked ? " disabled alpha-50" : "")}>
//                     {props.jobs.map((job) => (
//                         <Job key={job.name} cluster={props.name} jobData={job} />
//                     ))}
//                 </div>
//             )}
//         </div>
//     );
// }
