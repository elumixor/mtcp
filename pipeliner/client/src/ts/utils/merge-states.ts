import { useEffect, useState } from "react";

export function mergeStates<T>(...states: T[]) {
    const [result, setResult] = useState<T>();

    for (const state of states) useEffect(() => setResult(state), [state]);

    return [result, setResult] as const;
}
