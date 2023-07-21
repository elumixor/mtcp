export interface IJobData {
    name: string;
    description: string;
    clusters: string[];
}

export type IArtifactStatus = Record<string, boolean>;
