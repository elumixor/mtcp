import React from "react";
import { ReactSVG } from "react-svg";
import { JobRunState } from "responses";

export function StatusIcon({ status }: { status: "..." | "warn" | JobRunState }) {
    const cssStatus = status === "..." ? "retrieving" : status === "missing" ? "interrupted" : status;
    const className = "status-icon " + cssStatus;
    const statusText = status.replace("_", " ");
    const svg =
        status === "not_started" ? (
            <ReactSVG src="icons/not-started.svg" className={className} />
        ) : status === "done" ? (
            <ReactSVG src="icons/done.svg" className={className} />
        ) : status === "running" || status === "..." ? (
            <ReactSVG src="icons/running.svg" className={className} />
        ) : (
            <ReactSVG src="icons/warning.svg" className={className} />
        );

    return (
        <div className="status-container">
            {svg}
            {status === "..." ? (
                <span className="status-text retrieving">Retrieving status...</span>
            ) : (
                <span className={"status-text " + cssStatus}>{statusText}</span>
            )}
        </div>
    );
}
