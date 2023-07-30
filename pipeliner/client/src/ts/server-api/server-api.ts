import { useState } from "react";
import {
    IActionResponse,
    IConnectResponse,
    IGitSyncResponse,
    IJobData,
    IJobStatusResponse,
    ILogResponse,
    JobStatusesResponse,
} from "./responses";

export interface IServerApi {
    clusters(): string[];
    jobs(): IJobData[];
    connect(params: Params<{ cluster: string }>): IConnectResponse;
    git_sync(params?: Params<{ cluster?: string }>): IGitSyncResponse;
    job_status(params: Params<{ job: string }>): JobStatusesResponse;
    run_job(params: Params<{ cluster: string; job: string }>): IJobStatusResponse;
    interrupt_job(params: Params<{ cluster: string; job: string }>): IJobStatusResponse;
    delete_artifact(params: Params<{ cluster: string; job: string; artifact: string }>): IActionResponse;
    download_artifact(
        params: Params<{ cluster_from: string; cluster_to: string; job: string; artifact: string }>,
    ): IActionResponse;
    get_log(params: Params<{ cluster: string; file_path: string }>): ILogResponse;
}

// Adds the "debug" optional option
type Params<T> = T & { debug?: boolean };

const url = "http://localhost:8000";

export async function post<T extends keyof IServerApi>(endpoint: T, ...requestParams: Parameters<IServerApi[T]>) {
    try {
        const params = requestParams[0] ?? {};
        const response = await fetch(`${url}/${endpoint}`, {
            method: "POST",
            body: JSON.stringify(params),
        });

        if (!response.ok) throw new Error("Request failed");

        const data = await response.json();

        if (data.error) throw new Error(`Request failed. Error: ${data.error}`);

        console.log(`%c${endpoint}`, "color: #88aaff; font-weight: bold", params, data);

        return data as ReturnType<IServerApi[T]>;
    } catch (e) {
        console.log(`%c${endpoint}`, "color: #ff8888; font-weight: bold", requestParams[0], e);
        throw e;
    }
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
