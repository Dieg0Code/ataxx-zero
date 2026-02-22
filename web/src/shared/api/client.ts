type RequestOptions = {
  headers?: Record<string, string>;
};

async function resolveError(response: Response): Promise<Error> {
  const payload = (await response.json().catch(() => ({}))) as {
    message?: string;
    detail?: string;
  };
  return new Error(payload.message ?? payload.detail ?? `HTTP ${response.status}`);
}

export async function apiGet<T>(path: string, options?: RequestOptions): Promise<T> {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json", ...(options?.headers ?? {}) },
  });
  if (!response.ok) {
    throw await resolveError(response);
  }
  return (await response.json()) as T;
}

export async function apiPost<TResponse, TBody>(
  path: string,
  body: TBody,
  options?: RequestOptions,
): Promise<TResponse> {
  const response = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...(options?.headers ?? {}) },
    body: JSON.stringify(body),
  });
  if (!response.ok) {
    throw await resolveError(response);
  }
  return (await response.json()) as TResponse;
}

export async function apiPostNoContent<TBody>(
  path: string,
  body: TBody,
  options?: RequestOptions,
): Promise<void> {
  const response = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...(options?.headers ?? {}) },
    body: JSON.stringify(body),
  });
  if (!response.ok) {
    throw await resolveError(response);
  }
}
