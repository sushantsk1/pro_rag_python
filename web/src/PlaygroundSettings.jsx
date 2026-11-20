export const QUERY_METHODS = [['none', 'Query preparation · off'], ['rewrite', 'Rewrite question'], ['decompose', 'Decompose question'], ['hyde', 'HyDE passage']];

function Choice({ label, value, options, onChange }) {
  return <label className="lab-field lab-option-field"><span>{label}</span><select aria-label={label} value={value} onChange={event => onChange(event.target.value)}>{options.map(([id, name]) => <option key={id} value={id}>{name}</option>)}</select></label>;
}

export default function PlaygroundSettings({ stage, settings, update, options, document }) {
  if (stage === 'load') return <div className="lab-advanced-settings">
    <label className="lab-ocr-toggle"><input type="checkbox" checked={settings.ocr} disabled={settings.source !== 'document' || document?.format !== 'pdf'} onChange={event => update({ ocr: event.target.checked })} /><span>OCR scanned PDF pages</span></label>
    <p className="lab-method-note">Choose a library PDF. Pages without a text layer are read locally using RapidOCR; the library index stays unchanged.{options && !options.ocrAvailable ? ' Install the Python ocr extra to enable recognition.' : ''}</p>
  </div>;
  if (stage === 'embed') return <div className="lab-advanced-settings">
    <Choice label="Embedding model" value={settings.embeddingModel} options={(options?.embeddingModels || [{ id: 'configured', name: 'Configured local model' }]).map(model => [model.id, model.name])} onChange={embeddingModel => update({ embeddingModel, embedding: 'local' })} />
    <p className="lab-method-note">These are bi-encoders: the question and chunks are encoded separately. Each run builds a fresh index. New models download once, then run on CPU. E5 adds its required query and passage prefixes. Model token limits apply, so keep chunk sizes consistent when comparing.</p>
    <p className="lab-method-note">To compare models, run through Rerank, pin run A, select another model and rerun through Rerank. Compare timing, dimensions, retrieved passages and their order. Keep the source, chunk settings and question the same.</p>
  </div>;
  if (stage === 'query') return <div className="lab-advanced-settings"><Choice label="Query preparation" value={settings.queryPreparation} options={QUERY_METHODS.slice(1)} onChange={queryPreparation => update({ queryPreparation })} /><p className="lab-method-note">Adds one NVIDIA request before retrieval using the generation model and sampling settings. Decomposition combines results from up to three questions. HyDE searches with a hypothetical passage; that passage is never answer evidence.</p></div>;
  if (stage === 'rerank') return <div className="lab-advanced-settings">{settings.ranking === 'cross-encoder' && <Choice label="Cross-encoder model" value={settings.crossEncoderModel} options={(options?.crossEncoderModels || []).map(model => [model.id, model.name])} onChange={crossEncoderModel => update({ crossEncoderModel })} />}<p className="lab-method-note">Bi-encoder reranking uses the original question’s cosine similarity. A cross-encoder reads each question–passage pair together and scores relevance. It downloads once and runs locally; scores are raw logits, not probabilities.</p></div>;
  if (stage !== 'generate') return null;
  return <div className="lab-advanced-settings">
    <Choice label="NVIDIA model" value={settings.generationModel || ''} options={[["", 'Configured model' + (options ? ' · ' + options.defaultGenerationModel : '')], ...(options?.generationModels || []).map(model => [model.id, model.id])]} onChange={value => update({ generationModel: value || null })} />
    <p className="lab-method-note">NVIDIA API Catalog endpoints require an API key with model access. Trial availability and rate limits depend on your account.</p>
    <div className="lab-sampling-grid">{[['Temperature', 'temperature', 0, 1, 0.05], ['Top P', 'topP', 0.05, 1, 0.05], ['Max output tokens', 'maxTokens', 64, 2048, 16]].map(([label, key, min, max, step]) => <label key={key} className="lab-sampling"><span>{label}<output>{settings[key]}</output></span><input aria-label={label} type="range" min={min} max={max} step={step} value={settings[key]} onChange={event => update({ [key]: Number(event.target.value) })} /></label>)}</div>
    <label className="lab-field lab-option-field"><span>Token Top K</span><input aria-label="Token Top K" disabled value="Not exposed by these hosted endpoints" readOnly /></label>
    <p className="lab-method-note">Top P controls token sampling. Retrieval Top K controls the number of context passages. Adjust temperature or Top P individually to make comparisons easier.</p>
    <label className="lab-field lab-prompt-editor"><span>Answer instructions</span><textarea aria-label="Answer instructions" rows={5} maxLength={6000} value={settings.prompt} onChange={event => update({ prompt: event.target.value })} placeholder="For example: Answer concisely. Use bullet points when comparing policies." /></label>
    <p className="lab-method-note">Your instructions are followed by the required evidence, JSON and citation rules. Inspect the exact combined prompt in Input after a run.</p>
  </div>;
}
