import React, { useEffect, useState } from "react";
import { post } from "server-api";
import { Cluster } from "./Cluster";
import { IJobData } from "responses";

export function App() {
    const [clusters, setClusters] = useState<string[]>([]);
    const [jobs, setJobs] = useState<IJobData[]>([]);

    useEffect(() => {
        Promise.all([post<string[]>("clusters"), post<IJobData[]>("jobs")]).then(([clusters, jobs]) => {
            setClusters(clusters);
            setJobs(jobs);
        });
    }, []);

    return (
        <>
            <h1>
                Pipeliner
                <br />
                <button onClick={() => post("git_sync", { debug: true })}>Sync</button>
            </h1>
            <div id="clusters">
                {/* <Cluster name="[local]" key="[local]" jobs={[]} /> */}
                {clusters.map((cluster) => (
                    <Cluster name={cluster} jobs={jobs.filter((job) => job.clusters.includes(cluster))} key={cluster} />
                ))}
            </div>
        </>
    );
}
