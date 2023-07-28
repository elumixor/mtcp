export interface IJobData {
    name: string;
    description: string;
    clusters: string[];
    artifacts: string[];
}

export type IConnectResponse = { connected: boolean };
export type IGitSyncResponse = { status: string };

export type JobRunState = "not_started" | "running" | "done" | "interrupted" | "missing";
export type CondorRunState = "idle" | "running" | "done" | "hold";
export type ArtifactsResponse = Record<string, boolean>;

export interface IJobStatusResponse {
    artifacts?: ArtifactsResponse;
    status: JobRunState | "pending";
    condor?: {
        status: CondorRunState;
        id: number;
        log?: string;
    };
}

export type JobStatusesResponse = Record<string, IJobStatusResponse>;

export interface IActionResponse {
    success: boolean;
}
