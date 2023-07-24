import { useState } from "react";
import { post } from "./post";
import { ArtifactsResponse, IConnectResponse, IGitSyncResponse, IJobStatusResponse } from "responses";

export interface IServerApi {
    ["connect"]: [{ cluster: string }, IConnectResponse];
    ["git_sync"]: [{ cluster: string }, IGitSyncResponse];
    ["job_status"]: [{ cluster: string; job: string }, IJobStatusResponse];
    ["run_job"]: [{ cluster: string; job: string }, IJobStatusResponse];
    ["interrupt_job"]: [{ cluster: string; job: string }, IJobStatusResponse];
    ["delete_artifacts"]: [{ cluster: string; job: string }, ArtifactsResponse];
}

export function requests<TKeys extends (keyof IServerApi)[]>(...requests: getParams<TKeys>) {
    const [blocked, setBlocked] = useState(false);

    const states = requests.map(() => useState<unknown>());

    const calls = requests.map((request, i) => {
        const [, setState] = states[i];

        return async () => {
            setBlocked(true);
            setState(await post(request.endpoint, request.params));
            setBlocked(false);
        };
    });

    return [blocked, requests.map((_, i) => [states[i][0], calls[i]] as const)] as [boolean, getReturns<TKeys>];
}

type getParams<T extends (keyof IServerApi)[]> = {
    [K in keyof T]: {
        endpoint: T[K];
        params: IServerApi[T[K]][0];
    };
};

type getReturns<T extends (keyof IServerApi)[]> = {
    [K in keyof T]: [IServerApi[T[K]][1] | undefined, () => Promise<void>];
};
