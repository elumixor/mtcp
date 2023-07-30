export interface IJobData {
    name: string;
    description: string;
    clusters: string[];
    artifacts: string[];
}

export type IConnectResponse = { connected: boolean };
export type IGitSyncResponse = { status: string };

export type JobRunState = "not_started" | "running" | "done" | "interrupted" | "missing" | "failed" | "error";
export type CondorRunState = "idle" | "running" | "done" | "hold";
export type ArtifactsResponse = Record<string, boolean>;

export interface IJobStatusResponse {
    artifacts?: ArtifactsResponse;
    status?: JobRunState;
    condor?: {
        status: CondorRunState;
        id: number;
        log?: string;
    };
    log?: string;
    err?: string;
    out?: string;
}

export interface ILogResponse {
    success: boolean;
    error?: string;
    contents?: string;
    file_path?: string;
}

export type JobStatusesResponse = Record<string, IJobStatusResponse>;

export interface IActionResponse {
    success: boolean;
}
