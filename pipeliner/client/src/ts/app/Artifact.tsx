import React, { useContext, useState } from "react";
import { ClustersContext } from "./ClustersContext";
import { ReactSVG } from "react-svg";
import { post } from "server-api";
import { JobStatusesResponse } from "responses";

export function Artifact({
    artifact,
    exists,
    job,
    setJobStatus,
}: {
    artifact: string;
    job: string;
    exists: Record<string, boolean | null>;
    setJobStatus: React.Dispatch<React.SetStateAction<JobStatusesResponse | undefined>>;
}) {
    const clusters = useContext(ClustersContext);

    const availableCluster = Object.entries(exists).find(([, exists]) => exists === true)?.[0];

    return (
        <div className="artifact">
            <span className="artifact-name ellipsis">{artifact}</span>
            <span className="artifact-statuses">
                {clusters.map((cluster) => {
                    const clusterExists = exists[cluster];
                    const image =
                        clusterExists === null
                            ? null
                            : clusterExists
                            ? "trash"
                            : availableCluster
                            ? "download"
                            : "cross";
                    const actionAvailable = clusterExists !== null && (clusterExists || availableCluster);

                    const action = async () => {
                        if (clusterExists) {
                            const { success } = await post("delete_artifact", { artifact, cluster, job });

                            if (success)
                                setJobStatus((jobStatus) => {
                                    const newJobStatus = { ...jobStatus };

                                    const artifacts = newJobStatus[cluster].artifacts;
                                    if (artifacts) artifacts[artifact] = false;

                                    return newJobStatus;
                                });
                        } else if (availableCluster) {
                            const { success } = await post("download_artifact", {
                                artifact,
                                job,
                                cluster_to: cluster,
                                cluster_from: availableCluster,
                                debug: true,
                            });

                            if (success)
                                setJobStatus((jobStatus) => {
                                    const newJobStatus = { ...jobStatus };

                                    const artifacts = newJobStatus[cluster].artifacts;
                                    if (artifacts) artifacts[artifact] = true;

                                    return newJobStatus;
                                });
                        }
                    };

                    return (
                        <span
                            key={cluster}
                            className={
                                "artifact-exists " +
                                (clusterExists === null ? "pending" : clusterExists ? "good" : "neutral")
                            }
                        >
                            {clusterExists === null ? (
                                <></>
                            ) : (
                                <span
                                    className={"artifact-image" + (actionAvailable ? " button" : "")}
                                    onClick={action}
                                >
                                    <ReactSVG src={`assets/${image}.svg`} />
                                </span>
                            )}
                            {cluster}
                        </span>
                    );
                })}
            </span>
        </div>
    );
}
