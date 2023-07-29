import { createContext } from "react";
import { IJobData, IJobStatusResponse, JobStatusesResponse } from "server-api";

export const JobDataContext = createContext<IJobData>({
    name: "",
    description: "",
    artifacts: [],
    clusters: [],
});

export const JobStatusesContext = createContext<JobStatusesResponse>({});
