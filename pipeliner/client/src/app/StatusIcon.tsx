import React from "react";
import { ReactSVG } from "react-svg";

export function StatusIcon(props: { status: "..." | "not_started" | "done" | "warn" | "running" | "interrupted" }) {
    const cssStatus = props.status === "..." ? "retrieving" : props.status;
    const className = "status-icon " + cssStatus;
    const svg =
        props.status === "not_started" ? (
            <ReactSVG src="icons/not-started.svg" className={className} />
        ) : props.status === "done" ? (
            <ReactSVG src="icons/done.svg" className={className} />
        ) : props.status === "running" || props.status === "..." ? (
            <ReactSVG src="icons/running.svg" className={className} />
        ) : (
            <ReactSVG src="icons/warning.svg" className={className} />
        );

    return (
        <div className="status-container">
            {svg}
            {props.status === "..." ? (
                <span className="status-text retrieving">Retrieving status...</span>
            ) : (
                <span className={"status-text " + cssStatus}>{props.status}</span>
            )}
        </div>
    );
}
