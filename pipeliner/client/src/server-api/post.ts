const url = "http://localhost:8000";

export async function post<T = unknown>(endpoint: RequestInfo | URL, requestParams: Record<string, unknown> = {}) {
    const response = await fetch(`${url}/${endpoint}`, {
        method: "POST",
        body: JSON.stringify(requestParams),
    });

    if (!response.ok) throw new Error("Request failed");

    const data = await response.json();

    if (data.error) throw new Error(`Request failed. Error: ${data.error}`);

    return data as T;
}
