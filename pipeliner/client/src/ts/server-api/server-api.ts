import { useState } from "react";
import {
    ArtifactsResponse,
    IConnectResponse,
    IGitSyncResponse,
    IJobData,
    IJobStatusResponse,
    JobStatusesResponse,
} from "responses";

export interface IServerApi {
    jobs(): IJobData[];
    connect(params: Params<{ cluster: string }>): IConnectResponse;
    git_sync(params?: Params<{ cluster?: string }>): IGitSyncResponse;
    job_status(params: Params<{ job: string }>): JobStatusesResponse;
    run_job(params: Params<{ cluster: string; job: string }>): IJobStatusResponse;
    interrupt_job(params: Params<{ cluster: string; job: string }>): IJobStatusResponse;
    delete_artifacts(params: Params<{ cluster: string; job: string }>): ArtifactsResponse;
}

// Adds the "debug" optional option
type Params<T> = T & { debug?: boolean };

const url = "http://localhost:8000";

export async function post<T extends keyof IServerApi>(endpoint: T, ...requestParams: Parameters<IServerApi[T]>) {
    const params = requestParams[0] ?? {};
    const response = await fetch(`${url}/${endpoint}`, {
        method: "POST",
        body: JSON.stringify(params),
    });

    if (!response.ok) throw new Error("Request failed");

    const data = await response.json();

    if (data.error) throw new Error(`Request failed. Error: ${data.error}`);

    return data as ReturnType<IServerApi[T]>;
}

export function requests<TKeys extends (keyof IServerApi)[]>(...requests: getParams<TKeys>) {
    const [blocked, setBlocked] = useState(false);

    const states = requests.map(() => useState<unknown>());

    const calls = requests.map((request, i) => {
        const [, setState] = states[i];

        return async (...params: Parameters<IServerApi[TKeys[number]]>) => {
            setBlocked(true);
            setState(await post(request.endpoint, ...params));
            setBlocked(false);
        };
    });

    return [blocked, requests.map((_, i) => [states[i][0], calls[i]] as const)] as [boolean, getReturns<TKeys>];
}

type getParams<T extends (keyof IServerApi)[]> = {
    [K in keyof T]: {
        endpoint: T[K];
    };
};

type getReturns<T extends (keyof IServerApi)[]> = {
    [K in keyof T]: [
        ReturnType<IServerApi[T[K]]> | undefined,
        (...params: Parameters<IServerApi[T[K]]>) => Promise<void>,
    ];
};
