import assert from 'node:assert/strict';
import { test } from 'node:test';
import { streamPlayground } from '../../web/src/api.js';

const encoder = new TextEncoder();
function mocked(t, response) {
  const previous = globalThis.fetch;
  globalThis.fetch = async () => response;
  t.after(() => { globalThis.fetch = previous; });
}

test('decodes fragmented UTF-8 and NDJSON boundaries, delivering events in order', async t => {
  const events = [{ type:'stage_started',index:2,stage:{status:'running'} }, { type:'stage_progress',index:2,message:'Embedding café' }, { type:'run_completed',run:{id:'done'} }];
  const bytes = encoder.encode(events.map(e=>JSON.stringify(e)).join('\n')+'\n');
  mocked(t,new Response(new ReadableStream({start(c){for(let i=0;i<bytes.length;i+=3)c.enqueue(bytes.slice(i,i+3));c.close();}})));
  const received=[];
  const result=await streamPlayground({generate:true},{onEvent:e=>received.push(e)});
  assert.deepEqual(received,events);
  assert.equal(result.id,'done');
});

test('preserves earlier events and propagates a server failure', async t => {
  const received=[];
  mocked(t,new Response(JSON.stringify({type:'stage_completed',index:1,stage:{status:'complete'}})+'\n'+JSON.stringify({type:'run_error',error:{message:'Model unavailable',code:'EMBEDDING_UNAVAILABLE',status:503}})+'\n'));
  await assert.rejects(streamPlayground({},{onEvent:e=>received.push(e)}),e=>e.code==='EMBEDDING_UNAVAILABLE'&&e.status===503);
  assert.equal(received[0].type,'stage_completed');
});

test('interrupted stream never counts as a completed run', async t => {
  mocked(t,new Response('{"type":"stage_started","index":2}\n'));
  await assert.rejects(streamPlayground({},{onEvent:()=>{}}),e=>e.code==='STREAM_INTERRUPTED');
});

test('older backend gives a restart message instead of an empty map', async t => {
  mocked(t,new Response('{}',{status:404}));
  await assert.rejects(streamPlayground({},{onEvent:()=>{}}),e=>e.code==='BACKEND_UPDATE_REQUIRED'&&e.message.includes('Restart'));
});
