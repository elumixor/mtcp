import React from "react";

export function Artifact({ artifact, exists }: { artifact: string; exists: Record<string, boolean | null> }) {
    return (
        <div className="artifact">
            <span className="artifact-name ellipsis">{artifact}</span>
            <span>
                {Object.entries(exists).map(([cluster, exists]) => (
                    <span
                        key={cluster}
                        className={"artifact-exists " + (exists === null ? "pending" : exists ? "good" : "bad")}
                    >
                        {exists === null ? "..." : exists ? "Exists" : "Missing"}
                    </span>
                ))}
            </span>
        </div>
    );
}
