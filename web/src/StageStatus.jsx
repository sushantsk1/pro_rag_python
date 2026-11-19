export function formatDuration(value) {
  if (value == null) return 'Not run';
  if (value < 10) return Number(value).toFixed(2) + ' ms';
  if (value < 1000) return Math.round(value) + ' ms';
  if (value < 60000) return (value / 1000).toFixed(1) + ' s';
  return Math.floor(value / 60000) + 'm ' + Math.floor(value % 60000 / 1000) + 's';
}

export default function StageStatus({ entry, now, busy, dirty, className = '' }) {
  const state = dirty ? 'stale' : entry?.status || 'not_run';
  const running = state === 'running';
  const elapsed = running ? Math.max(0, now - entry.startedAtMs) : entry?.durationMs;
  const label = state === 'stale' ? 'Needs rerun' : running ? 'Running' : state === 'complete' ? 'Done' : state === 'error' ? 'Failed' : state === 'skipped' ? 'Skipped' : busy ? 'Waiting' : 'Not run';
  return <span className={'lab-stage-state lab-stage-' + state + ' ' + className} data-state={state}>
    {label}{elapsed != null && state !== 'stale' && <> · {formatDuration(elapsed)}</>}
  </span>;
}
