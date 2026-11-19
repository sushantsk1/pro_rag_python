import { useEffect, useRef, useState } from 'react';
import { Check, CircleAlert, LoaderCircle, Minus, Plus, RotateCcw } from 'lucide-react';
import StageStatus, { formatDuration } from './StageStatus.jsx';

export default function PlaygroundPipeline({ stages, settings, result, selected, dirty, busy, now, flowing, speed, onSelect, onDrop, chunks, onChunk }) {
  const viewport = useRef(null);
  const [width, setWidth] = useState(1000);
  const [zoom, setZoom] = useState(1);
  useEffect(() => {
    const observer = new ResizeObserver(entries => setWidth(entries[0].contentRect.width));
    observer.observe(viewport.current);
    return () => observer.disconnect();
  }, []);
  useEffect(() => {
    const node = viewport.current?.querySelectorAll('.lab-node')[selected];
    if (!node) return;
    const bounds = node.getBoundingClientRect(), view = viewport.current.getBoundingClientRect();
    if (bounds.right > view.right) viewport.current.scrollLeft += bounds.right - view.right + 18;
    else if (bounds.left < view.left) viewport.current.scrollLeft -= view.left - bounds.left + 18;
  }, [selected, zoom, width]);
  const sceneWidth = Math.max(880, width);
  const branchChunks = selected === 1 ? chunks.slice(0, 3) : [];
  const sceneHeight = branchChunks.length ? 200 : 150;
  return <div className={'lab-stage-canvas' + (flowing && !dirty ? ' lab-flowing' : '')} style={{ '--flow-duration': (2.2 / speed) + 's' }}>
    <div className="lab-stage-scroll" ref={viewport}>
      <div className="lab-stage-extent" style={{ width: sceneWidth * zoom, height: sceneHeight * zoom }}>
        <ol className="lab-flow" aria-label="RAG pipeline stages" style={{ width: sceneWidth, transform: 'scale(' + zoom + ')' }}>
          {stages.map((stage, index) => {
            const Icon = stage.icon, entry = result?.stages[index];
            const completed = !dirty && entry?.status === 'complete';
            const running = entry?.status === 'running' && !dirty;
            const waiting = busy && !running && !entry?.output;
            const hasNext = !dirty && result?.stages[index + 1]?.output;
            const method = stage.setting ? stage.methods.find(([key]) => key === settings[stage.setting])?.[1] : 'NVIDIA';
            const time = running ? 'Running' : formatDuration(entry?.durationMs);
            return <li key={stage.id}>
              <button type="button" className={'lab-node' + (selected === index ? ' lab-node-selected' : '') + (running ? ' lab-node-running' : '')} disabled={waiting} aria-busy={running} aria-pressed={selected === index} aria-label={stage.label + ' stage'} title={waiting ? 'Waiting for earlier stages to finish' : stage.label + ' · ' + (dirty ? 'Settings changed; rerun' : time)} onClick={() => onSelect(index)}
                onDragOver={event => { if (!busy) { event.preventDefault(); event.currentTarget.classList.add('lab-drop-target'); } }} onDragLeave={event => event.currentTarget.classList.remove('lab-drop-target')} onDrop={event => onDrop(event, index)}>
                <span className="lab-node-title"><Icon size={18} /><strong>{stage.label}</strong>{running ? <LoaderCircle size={14} className="spin lab-node-status" /> : entry?.status === 'error' && !dirty ? <CircleAlert size={16} className="lab-node-error" /> : completed ? <Check size={14} className="lab-node-status lab-node-check" /> : null}</span>
                <span className="lab-node-method">{method}</span>
                <StageStatus entry={entry} now={now} busy={busy} dirty={dirty} className="lab-node-progress" />
              </button>
              {index < stages.length - 1 && <div className={'lab-connector ' + (completed && hasNext ? 'lab-connector-ready' : '')} aria-hidden="true"><i className="lab-port" /><span /><i className="lab-port" />{[0, 1, 2].map(n => <b className="lab-packet" key={n} style={{ animationDelay: ((index * 0.24 + n * 0.2) / speed) + 's' }} />)}</div>}
              {index === 1 && branchChunks.length > 0 && <div className="lab-chunk-branch" aria-label="Chunk output preview"><span className="lab-branch-stem" /><div>{branchChunks.map(chunk => <button key={chunk.id} type="button" onClick={() => onChunk(chunk)}>{chunk.id}</button>)}</div></div>}
            </li>;
          })}
        </ol>
      </div>
    </div>
    <div className="lab-canvas-footer"><div className="lab-canvas-tools"><button type="button" className="icon-button" aria-label="Zoom pipeline out" disabled={zoom <= 0.75} onClick={() => setZoom(n => Math.max(0.75, n - 0.25))}><Minus size={15} /></button><span>{Math.round(zoom * 100)}%</span><button type="button" className="icon-button" aria-label="Zoom pipeline in" disabled={zoom >= 1.5} onClick={() => setZoom(n => Math.min(1.5, n + 0.25))}><Plus size={15} /></button><button type="button" className="icon-button" aria-label="Reset pipeline view" onClick={() => { setZoom(1); viewport.current.scrollTo({ left: 0 }); }}><RotateCcw size={15} /></button></div><small>{busy ? 'Live stage status · later stages wait' : flowing && !dirty ? 'Flow playback · measured outputs' : 'Select a stage to inspect'}</small></div>
  </div>;
}
