import { useEffect, useRef } from 'react';
import { X } from 'lucide-react';

export default function PlaygroundDetail({ chunk, parents, onClose }) {
  const dialog = useRef(null);
  useEffect(() => { dialog.current.showModal(); }, []);
  const parent = parents.find(p => p.id === chunk.parentId);
  return <dialog ref={dialog} className="lab-detail-dialog" onClose={onClose} aria-labelledby="chunk-detail-title">
    <header><div><small>{chunk.documentName}</small><h2 id="chunk-detail-title">{chunk.id}</h2></div><button type="button" className="icon-button" aria-label="Close chunk" onClick={() => dialog.current.close()}><X size={18} /></button></header>
    <div className="lab-detail-meta"><span>{chunk.text.length} characters</span><span>Offsets {chunk.start}–{chunk.end}</span>{chunk.location?.heading && <span>{chunk.location.heading}</span>}</div>
    <p className="lab-full-chunk">{chunk.text}</p>
    {parent && <details className="lab-extra"><summary>Parent context · {parent.id}</summary><p className="lab-full-chunk">{parent.text}</p></details>}
  </dialog>;
}
