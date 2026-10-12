// Thin fetch wrapper. All endpoints live under /api (Vercel Python function in production, uvicorn locally).

export class ApiError extends Error {}

function describe(detail: unknown): string {
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) return detail.map((d) => (typeof d === "object" && d ? (d as { msg?: string }).msg ?? JSON.stringify(d) : String(d))).join("; ");
  return "Request failed";
}

export async function api<T>(path: string, body?: unknown, signal?: AbortSignal): Promise<T> {
  let res: Response;
  try {
    res = await fetch(path, {
      method: body === undefined ? "GET" : "POST",
      headers: body === undefined ? undefined : { "Content-Type": "application/json" },
      body: body === undefined ? undefined : JSON.stringify(body),
      signal,
    });
  } catch (e) {
    if ((e as Error).name === "AbortError") throw e;
    throw new ApiError("Could not reach the server. Check your connection and try again.");
  }
  const text = await res.text();
  let json: unknown = null;
  try {
    json = text ? JSON.parse(text) : null;
  } catch {
    /* non-JSON body, handled below */
  }
  if (!res.ok) {
    if (json && typeof json === "object" && "detail" in json) throw new ApiError(describe((json as { detail: unknown }).detail));
    if (res.status === 504 || res.status === 408) throw new ApiError("The server took too long to answer. Try a smaller job.");
    if (res.status === 413) throw new ApiError("The request was too large for the server.");
    throw new ApiError(`Server error (${res.status}). Please try again.`);
  }
  if (json === null) throw new ApiError("The server returned an unexpected response.");
  return json as T;
}

export function errorText(e: unknown): string {
  return e instanceof Error ? e.message : String(e);
}
