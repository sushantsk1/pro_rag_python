# RAG Playground

This workbench lives inside the existing PRO RAG dashboard. Start the usual Python
backend and React frontend, then select **RAG Playground** in the sidebar. The direct
link is <http://127.0.0.1:5173/#playground>; no separate app or server is required.

## Try an experiment

1. Start with the sample policy documents, paste your own text, or choose a ready
   document from the active collection.
2. Select a compact stage card. One inspector opens below the workflow, with
   Output, Settings, and Input tabs; Embed and Retrieve also offer an Embedding map.
3. Choose an algorithm from the compact controls or open Settings to drag an
   algorithm card onto the matching pipeline stage. Change
   chunk size, overlap, retrieval count, or semantic boundary threshold as needed.
4. Choose **Run pipeline**, or **Run through** the selected stage. Results come from
   the Python backend. Changing a setting marks earlier output as stale until rerun.
5. Open transferred data to inspect the full payload, including vector values and
   IDs linking chunks, retrieval candidates, and context. Export a JSON snapshot.

The first visit automatically previews loading and recursive chunking. It does not
load the embedding model or call NVIDIA. Completing a vectorized run starts flow
playback along connections whose outputs exist. **Pause flow**, **Play flow**, and
the speed selector control this visualization; animation speed is separate from
measured server timings. Reduced-motion preferences disable moving particles.
The pipeline supports zoom and horizontal scrolling on small screens.

Stage cards show **Waiting**, **Running** with an elapsed timer, **Done** with the
server's measured duration, or **Failed**. Only the active stage spins; later stages
stay disabled until their inputs are ready. Completed stages can still be inspected
while the rest of the run continues. Semantic chunking performs its sentence
embeddings during Chunk, and that work is included in the Chunk duration.

Chunk output initially shows three excerpts. Page through them or select a chunk
to open its complete text, offsets, and parent context. The inspector can be closed
and reopened. Library documents are chosen from the active collection; switch that
collection in the Load stage's Settings tab.

## Embedding map

Run through **Embed** to create a map, or **Retrieve** to highlight retrieved
candidates. A full run opens Retrieve's map when starting from an earlier stage.
The map arrives immediately after Embed completes, before NVIDIA generation ends.
Its timing row separates vector creation from PCA projection, while the stage card
shows total Embed time (including index and map preparation).
Every dot corresponds to an actual run chunk; the question uses a separate diamond.
Selecting a dot or using the chunk dropdown shows its original text and full-vector
cosine similarity. **View chunk** opens the complete passage.

PCA is fitted on the chunk vectors, then the question is projected into the same
basis. Changing only the question does not rearrange document points. The map uses
up to three principal components; the 2D view pans and zooms, and the 3D view rotates
with dragging or arrow keys. Reset restores the camera. Positions are approximate:
retrieval continues to use the original vectors and algorithms, never chart distance.
The displayed percentage describes retained chunk variance, not retrieval accuracy.

Purple points are the first Top K candidates from the retrieval stage, before
reranking. In hybrid mode those candidates reflect RRF and may differ from cosine
order. With Embed-only runs no retrieval highlights are shown. The detail score is
cosine similarity even when the retrieval channel is keyword or hybrid.

Small or identical inputs can have fewer than two or three independent components;
the map explains overlapping points and the dropdown still selects each chunk.
TF–IDF questions with no vocabulary overlap have no query marker or cosine score.
Explicit stub configurations are labeled. PCA uses the existing NumPy dependency;
there is no additional model download or hosted visualization service.

## Available stages

| Stage | Options and visible results |
| --- | --- |
| Load | Keep sections or flatten text. Enable local RapidOCR for scanned PDF pages without a text layer. Inspect recognized text and OCR status. |
| Chunk | Fixed character windows; LangChain recursive splitting; parent–child splitting with parent context expansion; or adjacent sentence embedding similarity. Inspect boundaries, offsets, IDs, overlap, parents, and semantic split decisions. |
| Embed | MiniLM L6, smaller MiniLM L3, E5 Small v2, or corpus TF-IDF. Neural options are bi-encoders. E5 uses query/passage prefixes. Inspect actual vectors, dimensions, timings and PCA. |
| Query (optional) | Rewrite, decompose into up to three search questions, or generate a hypothetical HyDE passage. Adds one NVIDIA call plus search embedding time. Inspect generated search text. |
| Retrieve | BM25, FAISS cosine similarity, or reciprocal rank fusion of both (k=60). Inspect each channel and fused candidate scores. |
| Rerank | Fusion order, heuristic, original-question bi-encoder cosine, or local MiniLM L2/L6 cross-encoder relevance logits. Inspect incoming/outgoing order and selected context. |
| Generate | Select an NVIDIA Catalog model, edit answer instructions, and adjust temperature, Top P and output tokens. Required grounding and citation rules remain appended. Inspect exact prompts and request trace. |

Semantic chunking uses the selected local sentence embedding model,
including when TF–IDF is selected for the later vectorization stage. Similarity
below the threshold starts a new group; the character limit also forces a split.
Semantic grouping has no overlap. Parent–child uses parents up to four times the
child size, capped at 4,000 characters, and deduplicates expanded parents in context.

## Sources and comparison

Library experiments use the document's saved extracted sections, including their
existing chunk overlap. With OCR enabled, Load reparses the original PDF and runs local recognition on pages without extracted text. A failed scanned PDF can also be selected for OCR. Experiments never
write replacement chunks, vectors, evaluation results, or query history to the
library. This makes the workbench safe to explore alongside normal document queries.

**Pin run A** retains a result in the current page's memory. Change options and
rerun with the same original source and question to compare configurations, chunk
counts, selected evidence, and single-run server timings. Leaving the page clears
this comparison. Open **Compare** to view both runs. Export results if you want to
keep them; exports include full vectors and projection coordinates.

These timings are not a repeated latency benchmark, and ranking scores are not
accuracy measurements. Model initialization, corpus size, executed stages, and
generation affect timings. Use the existing evaluation feature or reproducible
retrieval report for labeled retrieval measurements.

## Bounds and generation

Runs use at most 40,000 source characters and 256 chunks. Library truncation is
displayed. Semantic grouping accepts at most 400 sentences per section. Excessive
chunk counts produce an error with guidance to increase size or reduce overlap.
Only one playground experiment executes at a time per API process.

Generation is included in full runs, without a checkbox. This sends the question
and selected evidence to the configured NVIDIA endpoint. Runs through earlier stages
remain local when query preparation is off. Query preparation makes one additional
NVIDIA call when enabled. Without a valid key, the active NVIDIA stage explains the error while
earlier local outputs remain available. Provider errors preserve the earlier stage
outputs. Missing embedding models report an error; they are never silently replaced
with fake vectors. Explicit development stub configurations are labeled in the UI.

The playground does not treat its heuristic ranking score as a probability that
the answer exists. When passages are retrieved, the grounded generator receives
them even if their score is low. This matters for questions such as a candidate's
name: the resume can state the name without containing the words "candidate" or
"name". Empty evidence still abstains, and citation validation remains enforced.

Generate's Input and Output tabs identify whether a model request was attempted
and whether a model response was received. Transferred data includes an `execution`
trace distinguishing no evidence, conflicts, missing credentials, a model abstention,
citation rejection, and a cited answer. A prepared `userPrompt` and `providerMode`
alone are not proof that a model request happened. The legacy Ask/evaluation
confidence policy is unchanged by this playground-specific behavior.

The JSON endpoint is `POST /api/v1/playground/run`. The dashboard uses
`POST /api/v1/playground/stream`, an NDJSON stream of actual stage start, progress,
completion, and error events. Both run the same pipeline. `untilStage` is zero-based
(0–5), retaining the original six-stage numbering. `untilStageId` can instead target
`load`, `chunk`, `embed`, `query`, `retrieve`, `rerank`, or `generate`; `query` requires
query preparation to be enabled. API clients can explicitly set `generate: false` for retrieval-only runs;
the dashboard presets keep it on. The schema is in <http://127.0.0.1:8787/docs>.

If embeddings are complete but map coordinates are absent, an older backend is
still running. Restart it and rerun Embed. The UI reports this mismatch explicitly.
For automatic reload during development, use the repository-root command:

~~~powershell
.\.venv\Scripts\python.exe -m uvicorn server.app:create_app --factory --reload --host 127.0.0.1 --port 8787
~~~

## Query preparation and model controls

Use the optional query selector above the pipeline to insert a Query card between Embed and Retrieve. Turning it off restores six stages. The original question is retained for reranking and answering. Decomposition retrieves for each question and combines rankings with RRF; HyDE uses its hypothetical passage vector for semantic search and the original question for BM25. Generated query text never becomes answer evidence.

Retrieve offers a search-query selector when query preparation is enabled. Each diamond and cosine score belongs to that actual search vector, projected into the original chunk PCA basis. Purple retrieval highlights reflect the fused results across all queries.

Model choices are limited to known IDs; keys and endpoint URLs stay on the server. Hosted model availability, trial credits and rate limits depend on the NVIDIA account. The current selected hosted endpoints do not expose a documented token Top K control, so it is disabled and rejected by the API instead of silently ignored. Retrieval Top K still selects context passages. Prompt editing changes additional answer instructions; evidence-only JSON and citation validation remain mandatory.

New embedding and cross-encoder weights download on first use to `data/models`, then run locally on CPU. One extra embedding model and one cross-encoder remain cached in memory. Changing the model always rebuilds the experiment vectors and FAISS index. For an embedding comparison, run through Rerank, pin A, change the embedding model, rerun through Rerank and open Compare. Keep chunk settings fixed and inspect timings, dimensions, evidence and order. Cold model load and download time are included; these are single runs, not accuracy benchmarks.

Install portable OCR dependencies using `.\.venv\Scripts\python.exe -m pip install -e ".[ocr]"`. RapidOCR uses ONNX Runtime on CPU; no separate Tesseract installation is required. Recognition defaults support English; results depend on scan quality. Large or slow PDFs still obey the parser timeout.
