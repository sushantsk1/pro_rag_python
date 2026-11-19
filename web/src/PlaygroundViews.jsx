import { BookOpen, Check, MessageSquare } from 'lucide-react';
const color = id => 'lab-tone-' + (((Number(String(id).replace(/\D/g, '')) || 1) - 1) % 3);

export function SourceDocuments({ documents, chunks = [] }) {
  return documents.map((doc, index) => <details className="lab-document" key={doc.id} open={index === 0}>
    <summary><BookOpen size={15} /><span>{doc.name}</span><small>{doc.sections.reduce((n, s) => n + s.text.length, 0)} characters</small></summary>
    <div className="lab-document-body">{doc.sections.map((section, sectionIndex) => {
      const pieces = chunks.filter(c => c.documentId === doc.id && c.section === sectionIndex).sort((a, b) => a.start - b.start);
      let cursor = 0;
      const spans = [];
      for (const piece of pieces) {
        if (piece.start > cursor) spans.push(section.text.slice(cursor, piece.start));
        const start = Math.max(cursor, piece.start);
        if (piece.end > start) spans.push(<mark className={color(piece.id)} key={piece.id} title={piece.id + ' · characters ' + piece.start + '–' + piece.end}>{section.text.slice(start, piece.end)}</mark>);
        cursor = Math.max(cursor, piece.end);
      }
      if (cursor < section.text.length) spans.push(section.text.slice(cursor));
      return <p key={sectionIndex}>{spans.length ? spans : section.text}</p>;
    })}</div>
  </details>);
}

export function Chunks({ chunks, parents = [], selectedId, onSelect }) {
  const byParent = Object.fromEntries(parents.map(p => [p.id, p]));
  return chunks.map(chunk => <article key={chunk.id} data-chunk-id={chunk.id} className={'lab-piece ' + color(chunk.id) + (selectedId === chunk.id ? ' lab-highlight' : '')}>
    <div className="lab-piece-head">{onSelect ? <button type="button" onClick={() => onSelect(chunk.id)} aria-pressed={selectedId === chunk.id}>{chunk.id}</button> : <strong>{chunk.id}</strong>}<small>{chunk.text.length} characters</small></div>
    <small>{chunk.documentName}</small><p>{chunk.text}</p>
    {chunk.parentId && <details className="lab-parent"><summary>Parent {chunk.parentId}</summary><p>{byParent[chunk.parentId]?.text}</p></details>}
  </article>);
}

export function VectorView({ output, selectedId, onSelect }) {
  const vector = output.vectors.find(v => v.chunkId === selectedId) || output.vectors[0];
  return <>
    <div className="lab-vector-meta"><strong>{output.model}</strong><span>{output.dimensions} dimensions · {output.vectors.length} vectors</span><small>{output.index} · normalized inner-product search</small></div>
    {output.mode === 'stub' && <p className="lab-warning">Test configuration: these are deterministic fixture vectors, not model embeddings.</p>}
    <label className="lab-field">Inspect a chunk<select value={vector?.chunkId || ''} onChange={e => onSelect(e.target.value)}>{output.vectors.map(row => <option key={row.chunkId}>{row.chunkId}</option>)}</select></label>
    {vector && <><div className="lab-vector-grid" aria-label={'Vector preview for ' + vector.chunkId}>{vector.values.slice(0, 48).map((value, i) => <span key={i} className={value < 0 ? 'negative' : ''} title={'Dimension ' + (i + 1) + (output.vocabulary ? ' · ' + output.vocabulary[i] : '') + ': ' + value}>{value.toFixed(2)}</span>)}</div>
      <small className="lab-vector-caption">First {Math.min(vector.values.length, 48)} of {vector.values.length} values · norm {vector.norm.toFixed(4)}. Full vectors are in the transferred data and export.</small></>}
  </>;
}

export function Candidates({ candidates, original = [], selected = [], selectedId }) {
  const originalRanks = new Map(original.map((c, i) => [c.id, i + 1]));
  const selectedIds = new Set(selected.map(c => c.id));
  if (!candidates.length) return <p className="lab-empty">No candidates matched this question.</p>;
  return candidates.map((candidate, index) => <article data-chunk-id={candidate.id} key={candidate.id} className={'lab-hit' + (selectedId === candidate.id ? ' lab-highlight' : '')}>
    <div className="lab-hit-head"><strong>#{index + 1} · {candidate.id || candidate.chunkId}</strong>{selectedIds.has(candidate.id) && <span className="lab-selected-label"><Check size={13} /> Context</span>}</div>
    <small>{candidate.documentName}{originalRanks.has(candidate.id) && originalRanks.get(candidate.id) !== index + 1 ? ' · moved from #' + originalRanks.get(candidate.id) : ''}</small>
    <p>{candidate.text || candidate.snippet}</p>
    {candidate.scores && <div className="lab-hit-scores">{Object.entries(candidate.scores).filter(([, value]) => value != null).map(([name, value]) => <span key={name}>{name}: {Number(value).toFixed(3)}</span>)}</div>}
  </article>);
}

export function GenerationTrace({ output }) {
  const trace = output.execution;
  if (!trace) return null;
  const explanations = {
    no_evidence: 'No model request: retrieval returned no evidence.',
    conflicting_evidence: 'No model request: conflicting evidence was detected.',
    missing_credentials: 'No model request: NVIDIA credentials are missing.',
    model_abstained: 'The model received the evidence and chose to abstain.',
    citation_validation_failed: 'The model replied, but its citations could not be validated.',
    answered: 'The model returned a cited answer.',
    disabled: 'Generation was not requested.',
    provider_error: 'The generation request did not complete successfully.',
  };
  return <div className="lab-generation-trace"><strong>{trace.fixture ? 'Test provider · no NVIDIA request' : trace.providerResponseReceived ? 'Model response received' : trace.providerRequestAttempted ? 'Model request attempted' : 'No model request'}</strong><span>{explanations[trace.decision] || 'Inspect the execution details in transferred data.'}</span></div>;
}

export function AnswerView({ output, onSource }) {
  if (output.error) return <><GenerationTrace output={output} /><div className="lab-error" role="alert"><strong>{output.error.code}</strong><p>{output.error.message}</p><small>The earlier pipeline outputs are still available.</small></div></>;
  if (output.answerStatus === 'disabled') return <div className="lab-generation-off"><MessageSquare size={26} /><h3>Your context is ready.</h3><p>Run the full pipeline to generate a grounded answer with NVIDIA.</p></div>;
  return <><GenerationTrace output={output} /><div className="lab-answer"><small>{output.answerStatus === 'abstained' ? 'Abstained' : 'Generated answer'}</small>{output.providerMode === 'stub' && <p className="lab-warning">Test provider: this is a fixture answer.</p>}<p>{output.answer}</p><div className="lab-citations">{output.citations.map(citation => <button type="button" key={citation.marker} onClick={() => onSource(citation.chunkId)}>{citation.marker} · {citation.chunkId}</button>)}</div></div></>;
}
