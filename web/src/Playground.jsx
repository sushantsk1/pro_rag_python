import { useEffect, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import {
  BookOpen, Check, ChevronLeft, ChevronRight, Code2, Copy, Download, FileInput,
  FolderOpen, GripVertical, ListFilter, LoaderCircle, MessageSquare, Pause,
  Play, Scissors, Search, X,
} from 'lucide-react';
import { streamPlayground } from './api.js';
import StageStatus, { formatDuration } from './StageStatus.jsx';
import EmbeddingMap from './EmbeddingMap.jsx';
import PlaygroundDetail from './PlaygroundDetail.jsx';
import PlaygroundPipeline from './PlaygroundPipeline.jsx';
import { AnswerView, Candidates, Chunks, GenerationTrace, SourceDocuments, VectorView } from './PlaygroundViews.jsx';
import './playground.css';

const STAGES = [
  { id: 'load', label: 'Load', icon: FileInput, setting: 'loader', methods: [['sections', 'Keep sections'], ['plain', 'Flatten text']] },
  { id: 'chunk', label: 'Chunk', icon: Scissors, setting: 'chunking', methods: [['fixed', 'Fixed size'], ['recursive', 'Recursive'], ['parent-child', 'Parent–child'], ['semantic', 'Semantic']] },
  { id: 'embed', label: 'Embed', icon: Code2, setting: 'embedding', methods: [['local', 'MiniLM'], ['tfidf', 'TF–IDF']] },
  { id: 'retrieve', label: 'Retrieve', icon: Search, setting: 'retrieval', methods: [['keyword', 'BM25'], ['semantic', 'Vector · FAISS'], ['hybrid', 'Hybrid · RRF']] },
  { id: 'rerank', label: 'Rerank', icon: ListFilter, setting: 'ranking', methods: [['none', 'Fusion order'], ['heuristic', 'Heuristic']] },
  { id: 'generate', label: 'Generate', icon: MessageSquare, setting: null, methods: [] },
];
const DEFAULT = {
  source: 'sample', text: '', documentId: null,
  question: 'What is the warranty for purchases after September 1, 2026?',
  loader: 'sections', chunking: 'recursive', chunkSize: 500, overlap: 80,
  semanticThreshold: 0.55, embedding: 'local', retrieval: 'hybrid',
  ranking: 'heuristic', topK: 6, generate: true,
};
const NAIVE = { loader: 'plain', chunking: 'fixed', chunkSize: 500, overlap: 0, embedding: 'local', retrieval: 'semantic', ranking: 'none', topK: 6, generate: true };
const ADVANCED = { loader: 'sections', chunking: 'recursive', chunkSize: 500, overlap: 80, embedding: 'local', retrieval: 'hybrid', ranking: 'heuristic', topK: 6, generate: true };
const NOTES = [
  'Keep source structure and locations, or flatten each document into plain text.',
  'Split the source into passages. Library chunks and indexes stay unchanged.',
  'Build normalized local vectors and inspect their values or projected positions.',
  'Find candidates with keyword search, vector search, or reciprocal rank fusion.',
  'Keep fusion order or apply the existing heuristic reranker. Inspect the final context.',
  'NVIDIA generates the answer from retrieved evidence. Inspect the exact prompts and citations.',
];
const duration = formatDuration;

function exportRun(run) {
  const url = URL.createObjectURL(new Blob([JSON.stringify(run, null, 2)], { type: 'application/json' }));
  const link = document.createElement('a');
  link.href = url; link.download = 'pro-rag-playground-' + run.id.slice(0, 8) + '.json'; link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function NumberSetting({ label, value, min, max, step = 1, onChange }) {
  const [draft, setDraft] = useState(String(value));
  useEffect(() => setDraft(String(value)), [value]);
  function commit(event) {
    const number = Number(event.target.value);
    const next = event.target.value === '' || !Number.isFinite(number) ? value : Math.max(min, Math.min(max, number));
    setDraft(String(next)); onChange(next);
  }
  return <label className="lab-number-setting"><span>{label}</span><input type="number" aria-label={'Playground ' + label.toLowerCase()} min={min} max={max} step={step} value={draft} onChange={event => {
    setDraft(event.target.value);
    if (event.target.value !== '' && event.target.validity.valid) onChange(event.target.valueAsNumber);
  }} onBlur={commit} /></label>;
}

function ChunkOutput({ chunks, boundaries, onInspect }) {
  const [page, setPage] = useState(0);
  useEffect(() => setPage(0), [chunks]);
  const lastPage = Math.max(0, Math.ceil(chunks.length / 3) - 1);
  const current = Math.min(page, lastPage);
  const first = current * 3;
  return <>
    <div className="lab-chunk-grid">{chunks.slice(first, first + 3).map(chunk => <button type="button" className="lab-chunk-preview" key={chunk.id} onClick={() => onInspect(chunk)}>
      <span className="lab-chunk-preview-head"><strong>{chunk.id}</strong><small>{chunk.text.length} chars</small></span>
      <p>{chunk.text}</p><small>{chunk.documentName}{chunk.parentId ? ' · ' + chunk.parentId : ''}</small>
    </button>)}</div>
    <div className="lab-output-footer"><div className="lab-pagination"><button type="button" className="icon-button" aria-label="Previous chunks" disabled={current === 0} onClick={() => setPage(n => n - 1)}><ChevronLeft size={17} /></button><span>{first + 1}–{Math.min(first + 3, chunks.length)} of {chunks.length}</span><button type="button" className="icon-button" aria-label="Next chunks" disabled={current >= lastPage} onClick={() => setPage(n => n + 1)}><ChevronRight size={17} /></button></div><small>Select a chunk to inspect its full text.</small></div>
    {boundaries.length > 0 && <details className="lab-extra"><summary>Semantic boundary decisions · {boundaries.length}</summary><div className="lab-boundary-list">{boundaries.map((boundary, i) => <p key={i}>{boundary.documentId} · offset {boundary.offset} · cosine {boundary.similarity.toFixed(3)} · {boundary.split ? 'split' : 'merge'}</p>)}</div></details>}
  </>;
}

export default function Playground({ collectionId, documents, collections, chooseCollection, config, navigate }) {
  const [settings, setSettings] = useState(DEFAULT);
  const [selected, setSelected] = useState(1);
  const [tab, setTab] = useState('output');
  const [inspectorOpen, setInspectorOpen] = useState(true);
  const [result, setResult] = useState(null);
  const [resultSettings, setResultSettings] = useState(null);
  const [baseline, setBaseline] = useState(null);
  const [comparisonOpen, setComparisonOpen] = useState(false);
  const [busy, setBusy] = useState(false);
  const [clock, setClock] = useState(Date.now());
  const [flowing, setFlowing] = useState(false);
  const [speed, setSpeed] = useState(0.75);
  const [selectedChunk, setSelectedChunk] = useState('C001');
  const [detailChunk, setDetailChunk] = useState(null);
  const [channel, setChannel] = useState('candidates');
  const [preset, setPreset] = useState('hybrid');
  const [error, setError] = useState('');
  const [message, setMessage] = useState('');
  const [rawOpen, setRawOpen] = useState(false);
  const [actionMount, setActionMount] = useState(null);
  const controller = useRef(null);
  const mounted = useRef(false);
  const readyDocuments = documents.filter(doc => doc.status === 'ready');
  const effective = { ...settings, generate: true, collectionId: settings.source === 'document' ? collectionId : null };
  const dirty = Boolean(result && JSON.stringify(effective) !== JSON.stringify(resultSettings));
  const stage = STAGES[selected], recorded = result?.stages[selected];
  const chunkOutput = result?.stages[1]?.output;
  const chunks = chunkOutput?.chunks || [], parents = chunkOutput?.parents || [];
  const embedding = result?.stages[2]?.output, retrieval = result?.stages[3]?.output;
  const activeStage = result?.stages.findIndex(entry => entry.status === 'running') ?? -1;
  const valid = settings.question.trim().length >= 2 && (settings.source !== 'text' || settings.text.trim().length >= 2) && (settings.source !== 'document' || readyDocuments.some(doc => doc.id === settings.documentId));
  const canCompare = Boolean(result?.stages[4]?.output) && !dirty && !busy;
  const comparable = baseline && result && baseline.sourceFingerprint === result.sourceFingerprint && baseline.configuration.question === result.configuration.question;
  const tabs = [...(selected === 2 || selected === 3 ? [['map', 'Embedding map']] : []), ['output', selected === 3 ? 'Results' : selected === 2 ? 'Vectors' : 'Output'], ['settings', 'Settings'], ['input', 'Input']];

  useEffect(() => {
    mounted.current = true; setActionMount(document.getElementById('playground-actions'));
    const timer = setTimeout(() => execute(1, DEFAULT), 0);
    return () => { clearTimeout(timer); mounted.current = false; controller.current?.abort(); };
  }, []);
  useEffect(() => {
    setSettings(previous => previous.source === 'document' ? { ...previous, source: 'sample', documentId: null } : previous);
  }, [collectionId]);
  useEffect(() => {
    if (!busy) return;
    setClock(Date.now());
    const timer = setInterval(() => setClock(Date.now()), 100);
    return () => clearInterval(timer);
  }, [busy]);

  function update(patch) {
    setSettings(previous => {
      const next = { ...previous, ...patch };
      if (next.chunking === 'semantic') next.overlap = 0;
      next.overlap = Math.min(next.overlap, next.chunkSize - 1);
      return next;
    });
    setPreset('custom'); setFlowing(false); setError(''); setMessage('');
  }
  function selectStage(index) {
    if (busy && result?.stages[index]?.status !== 'running' && !result?.stages[index]?.output) return;
    setSelected(index); setTab(index === 2 || index === 3 ? 'map' : 'output'); setRawOpen(false); setInspectorOpen(true);
  }
  async function execute(untilStage = 5, chosen = settings) {
    if (!mounted.current) return;
    controller.current?.abort();
    const requestController = new AbortController(); controller.current = requestController;
    const request = { ...chosen, generate: true, collectionId: chosen.source === 'document' ? collectionId : null };
    setBusy(true); setError(''); setMessage(''); setFlowing(false); setRawOpen(false);
    setResult(null); setResultSettings(request);
    try {
      const data = await streamPlayground({ ...request, untilStage }, {
        signal: requestController.signal,
        onEvent(event) {
          if (!mounted.current || requestController.signal.aborted) return;
          if (event.type === 'run_started') { setResult(event.run); return; }
          if (event.type === 'stage_started' || event.type === 'stage_completed' || event.type === 'stage_progress') {
            setResult(previous => {
              if (!previous) return previous;
              const stages = previous.stages.map((entry, index) => index !== event.index ? entry : event.stage || { ...entry, message: event.message });
              return { ...previous, stages };
            });
          }
          if (event.type === 'run_completed') setResult(event.run);
        },
      });
      if (!mounted.current || requestController.signal.aborted) return;
      setResult(data); setResultSettings(request); setInspectorOpen(true); setFlowing(untilStage >= 2);
      if (untilStage === 5 && selected < 3) { setSelected(3); setTab('map'); }
      else if (selected > untilStage) selectStage(untilStage);
      const firstMatch = data.stages[3]?.output?.candidates[0]?.id || data.stages[1]?.output?.chunks[0]?.id;
      if (firstMatch) setSelectedChunk(firstMatch);
      setMessage((data.stages[5].status === 'error' ? 'Generation needs attention' : 'Run complete') + ' · ' + data.summary.chunkCount + ' chunks · ' + duration(data.summary.durationMs) + ' server time');
    } catch (failure) {
      if (mounted.current && !requestController.signal.aborted) {
        const message = failure.message || 'The experiment could not run.';
        setError(message);
        setResult(previous => previous && ({
          ...previous, stages: previous.stages.map(entry => entry.status === 'running'
            ? { ...entry, status: 'error', durationMs: Math.max(0, Date.now() - entry.startedAtMs), message }
            : entry),
        }));
      }
    } finally {
      if (mounted.current && controller.current === requestController) setBusy(false);
    }
  }
  function chooseMethod(index, value) {
    const target = STAGES[index];
    if (busy || !target.setting || !target.methods.some(([id]) => id === value)) return;
    update({ [target.setting]: value });
  }
  function drop(event, index) {
    event.preventDefault(); event.currentTarget.classList.remove('lab-drop-target');
    if (busy) return;
    try {
      const choice = JSON.parse(event.dataTransfer.getData('application/pro-rag'));
      if (choice.stage !== index) { setMessage('Drop this algorithm onto its matching stage.'); return; }
      chooseMethod(index, choice.value); selectStage(index); setTab('settings');
    } catch { setMessage('Drag an algorithm from the Settings tab.'); }
  }
  function applyPreset(value) {
    setPreset(value);
    if (value !== 'custom') setSettings(previous => ({ ...previous, ...(value === 'naive' ? NAIVE : ADVANCED) }));
    setFlowing(false); setError('');
  }
  function sourceChoice(value) {
    update(value === 'sample' || value === 'text' ? { source: value, documentId: null } : { source: 'document', documentId: value });
  }
  function pin() {
    setBaseline(result); setMessage('Run A pinned. Change a setting and rerun with the same source and question.');
  }
  function tabKeys(event) {
    const index = tabs.findIndex(([id]) => id === tab);
    const next = event.key === 'ArrowRight' ? (index + 1) % tabs.length : event.key === 'ArrowLeft' ? (index + tabs.length - 1) % tabs.length : event.key === 'Home' ? 0 : event.key === 'End' ? tabs.length - 1 : -1;
    if (next >= 0) { event.preventDefault(); setTab(tabs[next][0]); event.currentTarget.querySelectorAll('[role=tab]')[next].focus(); }
  }
  function algorithmSelect() {
    return stage.setting && <label className="lab-field lab-method-select"><span className="sr-only">{stage.label} algorithm</span><select aria-label={stage.label + ' algorithm'} value={settings[stage.setting]} onChange={e => chooseMethod(selected, e.target.value)} disabled={busy}>{stage.methods.map(([value, label]) => <option key={value} value={value}>{label}</option>)}</select></label>;
  }
  function tuners() {
    return <>
      {algorithmSelect()}
      {selected === 1 && <><NumberSetting label="Size" value={settings.chunkSize} min={120} max={1500} onChange={value => update({ chunkSize: value })} />{settings.chunking !== 'semantic' ? <NumberSetting label="Overlap" value={settings.overlap} min={0} max={Math.min(200, settings.chunkSize - 1)} onChange={value => update({ overlap: value })} /> : <NumberSetting label="Threshold" value={settings.semanticThreshold} min={0} max={1} step={0.05} onChange={value => update({ semanticThreshold: value })} />}</>}
      {selected === 3 && <label className="lab-number-setting">Top K<select aria-label="Playground top K" value={settings.topK} disabled={busy} onChange={e => update({ topK: Number(e.target.value) })}>{[3, 5, 6, 10].map(n => <option key={n}>{n}</option>)}</select></label>}
      {selected === 5 && <span className="lab-generation-default">NVIDIA answer generation</span>}
    </>;
  }
  function inputView() {
    if (!recorded?.output) return <p className="lab-empty">Run through this stage to inspect its input.</p>;
    if (selected === 0) return <div className="lab-prose"><BookOpen size={22} /><h3>{result.configuration.source === 'sample' ? 'Sample documents' : result.configuration.source === 'text' ? 'Pasted text' : 'Library document'}</h3><p>{recorded.output.note}</p>{recorded.output.truncated && <p className="lab-warning">Using the first {recorded.output.characterLimit.toLocaleString()} of {recorded.output.availableCharacters.toLocaleString()} characters.</p>}</div>;
    if (selected === 1) return <SourceDocuments documents={result.stages[0].output.documents} chunks={chunks} />;
    if (selected === 2) return <div className="lab-chunk-grid"><Chunks chunks={chunks} parents={parents} selectedId={selectedChunk} onSelect={setSelectedChunk} /></div>;
    if (selected === 3) return <div className="lab-prose"><small>Question</small><h3>{result.configuration.question}</h3><p>{embedding.vectors.length} vectors · {embedding.dimensions} dimensions · {recorded.output.fusion}</p></div>;
    if (selected === 4) return <Candidates candidates={recorded.output.before} selectedId={selectedChunk} />;
    return <><GenerationTrace output={recorded.output} /><details className="lab-extra"><summary>System prompt</summary><pre>{recorded.output.systemPrompt}</pre></details><div className="lab-prompt"><small>Prepared question and evidence prompt</small><pre>{recorded.output.userPrompt}</pre></div></>;
  }
  function outputView() {
    if (!recorded?.output) return <div className="lab-empty"><h3>{recorded?.status === 'running' ? stage.label + ' is running…' : 'This stage is ready to explore.'}</h3><p>{recorded?.status === 'running' ? recorded.message : 'Run through ' + stage.label.toLowerCase() + ' to inspect its real output.'}</p></div>;
    const out = recorded.output;
    if (selected === 0) return <><p className="lab-source-note">{out.note}</p>{out.truncated && <p className="lab-warning">Source truncated to {out.characterLimit.toLocaleString()} characters.</p>}<SourceDocuments documents={out.documents} /></>;
    if (selected === 1) return <ChunkOutput chunks={chunks} boundaries={out.boundaries} onInspect={setDetailChunk} />;
    if (selected === 2) return <div className="lab-vector-output"><VectorView output={out} selectedId={selectedChunk} onSelect={setSelectedChunk} /></div>;
    if (selected === 3) {
      const hits = channel === 'candidates' ? out.candidates : out[channel].map(hit => ({ ...chunks.find(c => c.id === hit.chunkId), scores: { [channel === 'lexicalHits' ? 'BM25' : 'cosine']: hit.score } }));
      return <><label className="lab-field lab-channel-select">Candidate channel<select aria-label="Candidate channel" value={channel} onChange={e => setChannel(e.target.value)}><option value="candidates">Fused candidates ({out.candidates.length})</option><option value="lexicalHits">Keyword hits ({out.lexicalHits.length})</option><option value="semanticHits">Vector hits ({out.semanticHits.length})</option></select></label><div className="lab-results-list"><Candidates candidates={hits} /></div><p className="lab-map-note">Scores describe ranking, not the probability of a correct answer.</p></>;
    }
    if (selected === 4) return <><div className="lab-rank-columns"><section><h3>Incoming order</h3><Candidates candidates={out.before} selectedId={selectedChunk} /></section><section><h3>Selected context</h3><Candidates candidates={out.after} original={out.before} selected={out.selected} selectedId={selectedChunk} /></section></div><details className="lab-extra"><summary>Generation context · {out.sources.length} sources</summary><p>{out.note}</p>{out.sources.map(source => <article className="lab-context" key={source.chunkId}><strong>{source.chunkId}{source.parentId ? ' · ' + source.parentId : ''}</strong><p>{source.snippet}</p></article>)}</details></>;
    return <AnswerView output={out} onSource={id => { setSelectedChunk(id); selectStage(4); }} />;
  }
  function settingsView() {
    return <div className="lab-settings-view"><p>{NOTES[selected]}</p>
      <div className="lab-methods" aria-label={stage.label + ' algorithms'}>{stage.methods.map(([value, label]) => <button key={value} type="button" draggable={!busy} disabled={busy} className={'lab-method' + (settings[stage.setting] === value ? ' lab-method-selected' : '')} aria-pressed={settings[stage.setting] === value} onClick={() => chooseMethod(selected, value)} onDragStart={event => { event.dataTransfer.setData('application/pro-rag', JSON.stringify({ stage: selected, value })); event.dataTransfer.effectAllowed = 'copy'; }}><GripVertical size={15} /><span>{label}</span>{settings[stage.setting] === value && <Check size={15} />}</button>)}</div>
      {stage.methods.length > 0 && <small>Select an algorithm, or drag it onto its matching pipeline stage.</small>}
      {selected === 0 && collections?.length > 0 && <label className="lab-field lab-collection-setting">Active collection<select aria-label="Playground active collection" value={collectionId} onChange={e => chooseCollection(e.target.value)} disabled={busy}><option value="">Choose a collection</option>{collections.map(item => <option key={item.id} value={item.id}>{item.name}</option>)}</select></label>}
      {selected === 1 && <p className="lab-method-note">{settings.chunking === 'semantic' ? 'Adjacent sentence embeddings determine boundaries. Similarity below the threshold starts a new group; the size limit can also split it. Uses the local model with no overlap.' : settings.chunking === 'parent-child' ? 'Parents use up to four times the child size, capped at 4,000 characters. Retrieved children expand into deduplicated parent context.' : 'Sizes use characters. Recursive splitting prefers paragraph, sentence, and word boundaries; fixed splitting uses character windows.'}</p>}
      {selected === 2 && <p className="lab-method-note">MiniLM uses the configured local sentence model. TF–IDF uses corpus word frequencies and needs no model download. Semantic chunking still uses the local model.</p>}
      {selected === 5 && <><p>Full runs generate an answer automatically using NVIDIA. The question and selected evidence are sent to the configured endpoint. Earlier-stage runs remain local.</p>{!config?.nvidiaConfigured && config?.modelMode !== 'stub' && <button type="button" className="text-button" onClick={() => navigate('settings')}>Check generation settings</button>}</>}
    </div>;
  }

  const actions = <div className="lab-header-actions"><button type="button" className={'button lab-compare-button ' + (baseline ? 'secondary' : 'white')} disabled={!canCompare} title={baseline ? 'Compare with pinned run A' : 'Pin this result as run A'} onClick={() => { if (!baseline) pin(); setComparisonOpen(true); }}><Copy size={15} />Compare{baseline ? ' · A' : ''}</button><button className="button primary lab-run-pipeline" type="button" disabled={busy || !valid} onClick={() => execute(5)}>{busy ? <LoaderCircle size={16} className="spin" /> : <Play size={16} />}{busy ? 'Running…' : 'Run pipeline'}</button></div>;

  return <div className="rag-lab">
    {actionMount ? createPortal(actions, actionMount) : actions}
    <div className="lab-toolbar">
      <label className="lab-field lab-source-field"><span className="sr-only">Playground source</span><FolderOpen size={16} /><select aria-label="Playground source" value={settings.source === 'document' ? settings.documentId : settings.source} onChange={e => sourceChoice(e.target.value)} disabled={busy}><option value="sample">Sample documents</option><option value="text">Paste text</option>{readyDocuments.length > 0 && <optgroup label="Active collection">{readyDocuments.map(doc => <option key={doc.id} value={doc.id}>{doc.name}</option>)}</optgroup>}</select></label>
      <label className="lab-field"><span className="sr-only">Pipeline preset</span><select aria-label="Pipeline preset" value={preset} onChange={e => applyPreset(e.target.value)} disabled={busy}><option value="hybrid">Hybrid RAG</option><option value="naive">Naive RAG</option><option value="custom">Custom</option></select></label>
      <label className="lab-field lab-question"><span className="sr-only">Playground question</span><input aria-label="Playground question" maxLength={2000} value={settings.question} onChange={e => update({ question: e.target.value })} disabled={busy} /></label>
    </div>
    {settings.source === 'text' && <label className="lab-field lab-source-editor">Source text<textarea aria-label="Playground source text" rows={4} maxLength={40000} value={settings.text} onChange={e => update({ text: e.target.value })} disabled={busy} placeholder="Paste a document to experiment with…" /></label>}
    {error && <div className="lab-error" role="alert">{error}</div>}
    <div className="lab-flow-header"><div><strong>Pipeline</strong><span>6 steps</span>{dirty && <span className="lab-stale">Settings changed</span>}</div><div className="lab-playback-controls"><button type="button" disabled={busy || !result || dirty} onClick={() => setFlowing(value => !value)}>{flowing && !dirty ? <Pause size={15} /> : <Play size={15} />}{flowing && !dirty ? 'Pause flow' : 'Play flow'}</button><label><span className="sr-only">Flow playback speed</span><select aria-label="Flow playback speed" value={speed} onChange={e => setSpeed(Number(e.target.value))}>{[0.5, 0.75, 1, 1.5].map(n => <option key={n} value={n}>{n}×</option>)}</select></label></div></div>
    {busy && <div className="lab-live-progress" role="status" aria-live="polite"><LoaderCircle size={16} className="spin" /><span>{activeStage >= 0 ? STAGES[activeStage].label + ' · ' + result.stages[activeStage].message : 'Starting experiment…'}</span><span aria-live="off">{activeStage >= 0 ? duration(Math.max(0, clock - result.stages[activeStage].startedAtMs)) : ''}</span></div>}
    <PlaygroundPipeline stages={STAGES} settings={settings} result={result} selected={selected} dirty={dirty} busy={busy} now={clock} flowing={flowing} speed={speed} onSelect={selectStage} onDrop={drop} chunks={chunks} onChunk={setDetailChunk} />
    {inspectorOpen ? <section className="lab-inspector" aria-label={stage.label + ' inspector'}>
      <div className="lab-inspector-header"><h2><span>{String(selected + 1).padStart(2, '0')} /</span> {stage.label}</h2><div className="lab-tabs" role="tablist" aria-label="Stage views" onKeyDown={tabKeys}>{tabs.map(([id, label]) => <button type="button" key={id} id={'lab-tab-' + id} role="tab" aria-selected={tab === id} aria-controls="lab-tab-panel" tabIndex={tab === id ? 0 : -1} onClick={() => { setTab(id); setRawOpen(false); }}>{label}</button>)}</div><div className="lab-inspector-status"><StageStatus entry={recorded} now={clock} busy={busy} dirty={dirty} /><button type="button" className="icon-button" aria-label="Close inspector" onClick={() => setInspectorOpen(false)}><X size={17} /></button></div></div>
      {tab !== 'map' && <div className="lab-stage-controls"><fieldset disabled={busy}>{tuners()}</fieldset><button className="text-button lab-run-stage" type="button" disabled={busy || !valid} onClick={() => execute(selected)}>{busy ? <LoaderCircle size={15} className="spin" /> : <Play size={15} />}Run through {stage.label.toLowerCase()}</button></div>}
      <div id="lab-tab-panel" role="tabpanel" aria-labelledby={'lab-tab-' + tab} className={'lab-inspector-content lab-tab-' + tab}>
        {tab === 'map' ? <EmbeddingMap key={result?.id} embedding={embedding} entry={result?.stages[2]} now={clock} chunks={chunks} retrieval={retrieval} topK={result?.configuration.topK || settings.topK} selectedId={selectedChunk} onSelect={setSelectedChunk} onInspect={setDetailChunk} flowing={flowing && !dirty} /> : tab === 'settings' ? settingsView() : tab === 'input' ? inputView() : outputView()}
        {tab === 'map' && !embedding?.projection && <button className="text-button lab-run-stage" type="button" disabled={busy || !valid} onClick={() => execute(selected)}><Play size={15} />Run through {stage.label.toLowerCase()}</button>}
      </div>
      {recorded?.output && <details className="lab-raw" open={rawOpen} onToggle={event => setRawOpen(event.currentTarget.open)}><summary>Inspect transferred data</summary>{rawOpen && <pre>{JSON.stringify(recorded.output, null, 2)}</pre>}</details>}
    </section> : <button type="button" className="button secondary lab-reopen" onClick={() => setInspectorOpen(true)}>Inspect {stage.label.toLowerCase()}</button>}
    <div className="lab-bottom"><span role="status" aria-live="polite">{busy ? 'Live execution · later stages unlock when their inputs are ready' : dirty ? 'Settings changed · run again to update outputs' : message || 'Ready · NVIDIA generation included in full runs'}</span><div>{result && <><button className="text-button" type="button" disabled={!canCompare} onClick={pin}>Pin run A</button><button className="text-button" type="button" disabled={busy} onClick={() => exportRun(result)}><Download size={14} />Export run</button></>}</div></div>
    {comparisonOpen && baseline && <section className="lab-comparison" aria-label="Experiment comparison"><div className="lab-comparison-title"><h2>Compare experiments</h2><button type="button" className="icon-button" aria-label="Close comparison" onClick={() => setComparisonOpen(false)}><X size={17} /></button></div>{!comparable ? <p>Use the same source and question for a comparable second run.</p> : <><div className="lab-comparison-grid">{[['Run A', baseline], ['Current run', result]].map(([label, run]) => <div key={label}><strong>{label}</strong><p>{run.configuration.chunking} · {run.configuration.embedding} · {run.configuration.retrieval} · {run.configuration.ranking}</p><div className="lab-comparison-values"><span>{run.summary.chunkCount} chunks</span><span>{run.summary.sourceCount} sources</span><span>{duration(run.summary.durationMs)}</span></div><details className="lab-extra"><summary>Inspect evidence</summary>{run.sources.map(source => <p key={source.chunkId}>{source.documentName}: {source.snippet}</p>)}</details></div>)}</div><p className="lab-map-note">Single-run server timings include executed stages. This is a configuration comparison, not a repeated latency benchmark or accuracy score.</p></>}</section>}
    {detailChunk && <PlaygroundDetail chunk={detailChunk} parents={parents} onClose={() => setDetailChunk(null)} />}
  </div>;
}
