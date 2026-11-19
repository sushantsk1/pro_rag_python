import { useEffect, useMemo, useRef, useState } from 'react';
import { Minus, Plus, RotateCcw } from 'lucide-react';
import { formatDuration } from './StageStatus.jsx';

const HEIGHT = 330;
const score = value => value == null ? 'Unavailable' : value.toFixed(3);

export default function EmbeddingMap({ embedding, entry, now, chunks, retrieval, topK, selectedId, onSelect, onInspect, flowing }) {
  const [mode, setMode] = useState('2d');
  const [camera, setCamera] = useState({ yaw: -0.5, pitch: 0.35, zoom: 1, x: 0, y: 0 });
  const [hovered, setHovered] = useState(null);
  const pointer = useRef(null);
  const plot = useRef(null);
  const [plotWidth, setPlotWidth] = useState(760);
  const projection = embedding?.projection;
  useEffect(() => {
    if (!plot.current) return;
    const observer = new ResizeObserver(entries => setPlotWidth(entries[0].contentRect.width));
    observer.observe(plot.current);
    return () => observer.disconnect();
  }, [Boolean(projection)]);
  const retrieved = useMemo(() => new Map((retrieval?.candidates || []).slice(0, topK).map((c, i) => [c.id, i + 1])), [retrieval, topK]);
  const byId = useMemo(() => new Map(chunks.map(c => [c.id, c])), [chunks]);
  const points = projection?.points || [];
  const active = points.find(p => p.chunkId === selectedId) || points[0];
  const chunk = byId.get(active?.chunkId);
  const reset = () => setCamera({ yaw: -0.5, pitch: 0.35, zoom: 1, x: 0, y: 0 });

  if (!projection) {
    if (embedding?.vectors?.length) return <div className="lab-empty lab-map-missing" role="alert"><h3>Embeddings are ready; map coordinates are missing.</h3><p>The backend is returning an older response. Restart the Python backend and rerun Embed to build this map.</p></div>;
    if (entry?.status === 'running') return <div className="lab-empty" role="status"><h3>Creating your embedding space…</h3><p>{entry.message}</p><small>Elapsed {formatDuration(Math.max(0, now - entry.startedAtMs))}. Retrieval will wait until vectors and map coordinates are ready.</small></div>;
    if (entry?.status === 'error') return <div className="lab-empty" role="alert"><h3>Embedding did not finish.</h3><p>{entry.message || 'Retry the run after checking the local model.'}</p></div>;
    return <div className="lab-empty"><h3>See your embeddings in space.</h3><p>Run through Embed to build the map from actual chunk vectors. Run through Retrieve to highlight matches.</p></div>;
  }

  const WIDTH = Math.max(280, plotWidth);
  const BOX = { left: 62, top: 23, width: WIDTH - 84, height: 250 };

  const query = projection.query;
  const all = [...points.map(p => p.coordinates), ...(query.hasSignal ? [query.coordinates] : [])];
  const extent = Math.max(0.05, ...all.map(p => Math.hypot(...p))) * 1.15;
  const unit = (Math.min(BOX.width, BOX.height) / 2) / extent * camera.zoom;
  const limits = [0, 1].map(axis => {
    const values = all.map(point => point[axis]);
    const min = Math.min(...values), max = Math.max(...values);
    return { center: (min + max) / 2, span: Math.max(0.08, max - min) * 1.3 };
  });
  const unitX = mode === '2d' ? BOX.width / limits[0].span * camera.zoom : unit;
  const unitY = mode === '2d' ? BOX.height / limits[1].span * camera.zoom : unit;
  function position(coordinates) {
    let [x, y, z] = coordinates;
    if (mode === '3d') {
      const rx = x * Math.cos(camera.yaw) + z * Math.sin(camera.yaw);
      const rz = -x * Math.sin(camera.yaw) + z * Math.cos(camera.yaw);
      x = rx; z = y * Math.sin(camera.pitch) + rz * Math.cos(camera.pitch);
      y = y * Math.cos(camera.pitch) - rz * Math.sin(camera.pitch);
    }
    return { x: BOX.left + BOX.width / 2 + (x - (mode === '2d' ? limits[0].center : 0)) * unitX + camera.x, y: BOX.top + BOX.height / 2 - (y - (mode === '2d' ? limits[1].center : 0)) * unitY + camera.y, z };
  }
  const plotted = points.map(p => ({ ...p, ...position(p.coordinates) })).sort((a, b) => a.z - b.z);
  const question = position(query.coordinates);
  const ticks = [-2, -1, 0, 1, 2];
  const origin = position([0, 0, 0]);
  const displayPoint = plotted.find(p => p.chunkId === (hovered || active?.chunkId));
  const variance = projection.explainedVariance.slice(0, mode === '2d' ? 2 : 3).reduce((sum, n) => sum + n, 0);
  const available = projection.componentsAvailable;

  function pointerDown(event) {
    if (event.button !== 0 || event.target.closest('[data-map-point]')) return;
    pointer.current = { x: event.clientX, y: event.clientY, camera: { ...camera } };
    event.currentTarget.setPointerCapture(event.pointerId);
  }
  function pointerMove(event) {
    if (!pointer.current) return;
    const start = pointer.current;
    const dx = event.clientX - start.x, dy = event.clientY - start.y;
    const scale = WIDTH / event.currentTarget.getBoundingClientRect().width;
    setCamera(mode === '3d'
      ? { ...start.camera, yaw: start.camera.yaw + dx * 0.009, pitch: Math.max(-1.3, Math.min(1.3, start.camera.pitch + dy * 0.009)) }
      : { ...start.camera, x: start.camera.x + dx * scale, y: start.camera.y + dy * scale });
  }
  function keyPoint(event, id) {
    if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); onSelect(id); }
  }
  function orbitKey(event) {
    if (event.target !== event.currentTarget) return;
    const direction = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1] }[event.key];
    if (!direction) return;
    event.preventDefault();
    setCamera(previous => mode === '3d'
      ? { ...previous, yaw: previous.yaw + direction[0] * 0.15, pitch: Math.max(-1.3, Math.min(1.3, previous.pitch + direction[1] * 0.15)) }
      : { ...previous, x: previous.x + direction[0] * 18, y: previous.y + direction[1] * 18 });
  }
  const changeZoom = amount => setCamera(previous => ({ ...previous, zoom: Math.max(0.5, Math.min(3, previous.zoom + amount)) }));

  return <div className={'lab-map ' + (flowing ? 'lab-map-flowing' : '')}>
    <div className="lab-map-toolbar">
      <span>{embedding.mode === 'tfidf' ? 'TF–IDF' : embedding.model.split('/').pop()} · {embedding.dimensions} dimensions · PCA projection</span>
      <div className="lab-map-legend"><span><i />Chunks</span><span><i className="retrieved" />{retrieval ? 'Retrieved' : 'No retrieval yet'}</span><span><i className="query" />Question</span></div>
      <div className="lab-mode" role="group" aria-label="Map dimensions">{['2d', '3d'].map(value => <button type="button" key={value} aria-pressed={mode === value} onClick={() => { setMode(value); reset(); }}>{value.toUpperCase()}</button>)}</div>
    </div>
    {embedding.embeddingDurationMs != null && <p className="lab-embedding-timings">Embeddings done · {formatDuration(embedding.embeddingDurationMs)}<span>Map projection · {formatDuration(embedding.projectionDurationMs)}</span></p>}
    {embedding.mode === 'stub' && <p className="lab-warning">Test configuration: this map uses fixture vectors, not model embeddings.</p>}
    <div className="lab-map-layout"><div className="lab-map-plot" ref={plot}>
      <svg className={'lab-scatter lab-scatter-' + mode} viewBox={'0 0 ' + WIDTH + ' ' + HEIGHT} role="group" aria-label={mode === '3d' ? 'Rotatable 3D PCA embedding map. Use arrow keys to rotate.' : '2D PCA embedding map. Use arrow keys to pan.'} tabIndex={0}
        onPointerDown={pointerDown} onPointerMove={pointerMove} onPointerUp={() => { pointer.current = null; }} onPointerCancel={() => { pointer.current = null; }} onKeyDown={orbitKey}>
        <defs><clipPath id="embedding-plot-clip"><rect x={BOX.left} y={BOX.top} width={BOX.width} height={BOX.height} /></clipPath></defs>
        {mode === '2d' ? <>
          {ticks.map(n => {
            const x = BOX.left + BOX.width / 2 + n * BOX.width / 4;
            const y = BOX.top + BOX.height / 2 + n * BOX.height / 4;
            const xv = (x - origin.x) / unitX, yv = (origin.y - y) / unitY;
            return <g className="lab-map-grid" key={n}><line x1={x} x2={x} y1={BOX.top} y2={BOX.top + BOX.height} /><line x1={BOX.left} x2={BOX.left + BOX.width} y1={y} y2={y} /><text x={x} y={BOX.top + BOX.height + 22} textAnchor="middle">{xv.toFixed(1)}</text><text x={BOX.left - 12} y={y + 4} textAnchor="end">{yv.toFixed(1)}</text></g>;
          })}
          <path className="lab-map-axis" d={'M' + BOX.left + ' ' + BOX.top + 'V' + (BOX.top + BOX.height) + 'H' + (BOX.left + BOX.width)} />
          <text className="lab-axis-label" x={BOX.left + BOX.width / 2} y={HEIGHT - 9} textAnchor="middle">PCA 1 (X)</text>
          <text className="lab-axis-label" transform={'translate(17 ' + (BOX.top + BOX.height / 2) + ') rotate(-90)'} textAnchor="middle">PCA 2 (Y)</text>
        </> : <g className="lab-map-axes-3d">{[0, 1, 2].map(axis => {
          const start = [0, 0, 0], end = [0, 0, 0]; start[axis] = -extent * 0.85; end[axis] = extent * 0.85;
          const a = position(start), b = position(end);
          return <g key={axis}><line x1={a.x} y1={a.y} x2={b.x} y2={b.y} /><text x={b.x + 6} y={b.y - 7}>{'PCA ' + (axis + 1)}</text></g>;
        })}</g>}
        <g clipPath="url(#embedding-plot-clip)">
          {query.hasSignal && plotted.filter(p => retrieved.has(p.chunkId)).map(p => <line className="lab-map-link" key={p.chunkId} x1={question.x} y1={question.y} x2={p.x} y2={p.y} />)}
          {plotted.map(point => <g key={point.chunkId} className={'lab-map-dot ' + (retrieved.has(point.chunkId) ? 'is-retrieved ' : '') + (active?.chunkId === point.chunkId ? 'is-selected' : '')}
            data-map-point={point.chunkId} role="button" tabIndex={0} aria-label={point.chunkId + ', ' + byId.get(point.chunkId)?.documentName + ', cosine similarity ' + score(point.cosineSimilarity)} aria-pressed={active?.chunkId === point.chunkId}
            onClick={() => onSelect(point.chunkId)} onKeyDown={e => keyPoint(e, point.chunkId)} onMouseEnter={() => setHovered(point.chunkId)} onMouseLeave={() => setHovered(null)}
            transform={'translate(' + point.x + ' ' + point.y + ')'}>
            <title>{point.chunkId + ' · ' + byId.get(point.chunkId)?.documentName + ' · cosine ' + score(point.cosineSimilarity)}</title>
            <circle className="lab-point-target" r={12} /><circle className="lab-point-ring" r={10} /><circle className="lab-point-center" r={active?.chunkId === point.chunkId ? 5.5 : 4.5} />
          </g>)}
          {query.hasSignal && <g className="lab-query-point" transform={'translate(' + question.x + ' ' + question.y + ')'}><path d="M0 -7L7 0L0 7L-7 0Z" /><text x={-12} y={-13} textAnchor="end">Your question</text></g>}
        </g>
        {displayPoint && <g className="lab-map-tooltip" pointerEvents="none" transform={'translate(' + Math.max(BOX.left + 10, Math.min(BOX.left + BOX.width - 95, displayPoint.x + 12)) + ' ' + Math.max(BOX.top + 18, Math.min(BOX.top + BOX.height - 10, displayPoint.y - 12)) + ')'}><rect x={-7} y={-17} width={79} height={27} rx={5} /><text>{displayPoint.chunkId}</text></g>}
      </svg>
      <div className="lab-map-navigation"><span>{mode === '3d' ? 'Drag or use arrow keys to rotate' : 'Drag or use arrow keys to pan'}</span><div>
        <button type="button" className="icon-button" aria-label="Zoom map out" onClick={() => changeZoom(-0.25)} disabled={camera.zoom <= 0.5}><Minus size={15} /></button><span>{Math.round(camera.zoom * 100)}%</span><button type="button" className="icon-button" aria-label="Zoom map in" onClick={() => changeZoom(0.25)} disabled={camera.zoom >= 3}><Plus size={15} /></button><button type="button" className="icon-button" aria-label="Reset map view" onClick={reset}><RotateCcw size={15} /></button>
      </div></div>
    </div><aside className="lab-map-detail" aria-label="Selected chunk details">
      <span className="lab-detail-eyebrow">Selected chunk</span>
      <label className="lab-field"><span className="sr-only">Select a map chunk</span><select aria-label="Select a map chunk" value={active?.chunkId || ''} onChange={e => onSelect(e.target.value)}>{points.map(p => <option key={p.chunkId}>{p.chunkId}</option>)}</select></label>
      <p className="lab-map-filename">{chunk?.documentName}</p><p className="lab-map-excerpt">{chunk?.text}</p>
      <div className="lab-similarity"><span>Cosine similarity</span><strong>{score(active?.cosineSimilarity)}</strong></div>
      <small>Uses the full {embedding.dimensions}D vectors.{retrieved.has(active?.chunkId) ? ' Retrieved #' + retrieved.get(active.chunkId) + '.' : ''}</small>
      {chunk && <button type="button" className="text-button" onClick={() => onInspect(chunk)}>View chunk</button>}
    </aside></div>
    <p className="lab-map-note">Positions are a PCA projection; retrieval uses the full vectors. {Math.round(variance * 100)}% of chunk variance shown.{available < (mode === '2d' ? 2 : 3) ? ' Only ' + available + ' independent component' + (available === 1 ? '' : 's') + '; some points may overlap.' : ''}{!query.hasSignal ? ' The question has no vocabulary overlap; its marker and cosine scores are unavailable.' : ''}</p>
  </div>;
}
