import React, { useEffect, useState } from "react";
import { post } from "server-api";
import { Cluster } from "./Cluster";
import { IJobData } from "responses";

export function App() {
    const [clusters, setClusters] = useState<string[]>([]);
    const [jobs, setJobs] = useState<IJobData[]>([]);

    const update = async () => {
        const [clusters, jobs] = await Promise.all([post<string[]>("clusters"), post<IJobData[]>("jobs")]);

        setClusters(clusters);
        setJobs(jobs);
    };

    const sync = async () => {
        setClusters([]);
        setJobs([]);

        const status = await post("git_sync", { debug: true });
        console.log(status);

        await update();
    };

    useEffect(() => void update(), []);

    return (
        <>
            <h1>
                Pipeliner
                <br />
                <button onClick={sync}>Sync</button>
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
