import React from "react";
import { ReactSVG } from "react-svg";

export function IconButton({
    image,
    disabled = false,
    className = "",
    onClick,
}: {
    image: string;
    disabled?: boolean;
    className?: string;
    onClick?: () => unknown;
}) {
    return (
        <span className={className + " icon-button" + (disabled ? " disabled" : " button")} onClick={onClick}>
            <ReactSVG src={`assets/${image}.svg`} />
        </span>
    );
}
