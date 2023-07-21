import React, { useEffect, useState } from "react";
import { post } from "utils";
import { Clusters } from "./Clusters";
import { IJobData } from "responses";

export function App() {
    const [clusters, setClusters] = useState([]);
    const [jobs, setJobs] = useState<IJobData[]>([]);

    useEffect(() => {
        Promise.all([
            post("/clusters"),
            post("/jobs"),
        ]).then(([clusters, jobs]) => {
            setClusters(clusters);
            setJobs(jobs);
        });
    }, []);

    return <>
        <h1>Pipeliner</h1>
        <Clusters clusters={clusters} jobs={jobs}/>
    </>
}