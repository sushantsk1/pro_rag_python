const ROOT = '/api/v1';

export class ApiError extends Error {
  constructor(message, code, status, details = {}, requestId = '') {
    super(message);
    this.name = 'ApiError';
    this.code = code;
    this.status = status;
    this.details = details;
    this.requestId = requestId;
  }
}

export async function api(path, options = {}) {
  const { body, ...rest } = options;
  const isForm = body instanceof FormData;
  let response;
  try {
    response = await fetch(`${ROOT}${path}`, {
      ...rest,
      headers: {
        ...(body && !isForm ? { 'Content-Type': 'application/json' } : {}),
        ...(rest.headers || {})
      },
      body: body == null ? undefined : isForm ? body : JSON.stringify(body)
    });
  } catch {
    throw new ApiError('Cannot reach the local API. Start the backend and try again.', 'NETWORK_ERROR', 0);
  }
  if (response.status === 204) return null;
  const payload = await response.json().catch(() => null);
  if (!response.ok) {
    const problem = payload?.error || {};
    throw new ApiError(
      problem.message || `Request failed (HTTP ${response.status}).`,
      problem.code || 'REQUEST_FAILED',
      response.status,
      problem.details || {},
      problem.requestId || ''
    );
  }
  return payload;
}

export async function streamPlayground(body, { signal, onEvent }) {
  let response;
  try {
    response = await fetch(ROOT + '/playground/stream', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body), signal,
    });
  } catch {
    throw new ApiError('Cannot reach the local API. Start the backend and try again.', 'NETWORK_ERROR', 0);
  }
  if (response.status === 404) {
    throw new ApiError('The backend is running an older version. Restart it to enable embedding maps and live stage progress, then run again.', 'BACKEND_UPDATE_REQUIRED', 404);
  }
  if (!response.ok) {
    const payload = await response.json().catch(() => null);
    throw new ApiError(payload?.error?.message || 'The experiment could not start.', payload?.error?.code || 'REQUEST_FAILED', response.status);
  }
  if (!response.body) throw new ApiError('This browser could not read live progress.', 'STREAM_UNAVAILABLE', 0);
  const reader = response.body.getReader(), decoder = new TextDecoder();
  let buffer = '', completed = null;
  function consume(line) {
    if (!line.trim()) return;
    const event = JSON.parse(line);
    onEvent(event);
    if (event.type === 'run_completed') completed = event.run;
    if (event.type === 'run_error') throw new ApiError(event.error.message, event.error.code, event.error.status);
  }
  try {
    for (;;) {
      const { value, done } = await reader.read();
      buffer += decoder.decode(value, { stream: !done });
      let newline;
      while ((newline = buffer.indexOf('\n')) >= 0) {
        const line = buffer.slice(0, newline); buffer = buffer.slice(newline + 1); consume(line);
      }
      if (done) break;
    }
    if (buffer.trim()) consume(buffer);
    if (!completed) throw new ApiError('The connection ended before the run completed. Completed stage outputs are still available; retry the run.', 'STREAM_INTERRUPTED', 0);
    return completed;
  } finally {
    await reader.cancel().catch(() => {});
    reader.releaseLock();
  }
}

export function queryString(params) {
  const q = new URLSearchParams();
  Object.entries(params).forEach(([key, value]) => {
    if (value !== null && value !== undefined && value !== '') q.set(key, String(value));
  });
  const s = q.toString();
  return s ? `?${s}` : '';
}
