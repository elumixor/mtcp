import React from "react";

export function Artifact(props: { artifact: string; exists: boolean }) {
    return (
        <div className="artifact flex">
            <span className="artifact-name ellipsis">{props.artifact}</span>
            <span className={"artifact-exists " + (props.exists ? "good" : "bad")}>
                {props.exists ? "Exists" : "Missing"}
            </span>
        </div>
    );
}
