import React, {useEffect, useRef, useState, useCallback} from 'react';
import {createRoot} from 'react-dom/client';
import {Camera, CameraOff, Shirt, RefreshCw, Activity, Wifi, WifiOff, Scan, Sparkles, CircleHelp, Check, AlertTriangle} from 'lucide-react';
import './style.css';

const qApi=new URLSearchParams(window.location.search).get('api');
if(qApi)localStorage.setItem('arw_api',qApi);
const API=(qApi||localStorage.getItem('arw_api')||import.meta.env.VITE_API_URL||(import.meta.env.PROD&&!window.location.hostname.endsWith('github.io')?window.location.origin:'http://localhost:8000')).replace(/\/$/,'');
const WS=API.replace(/^http/,'ws')+'/ws/tryon';
const outfits=[{id:'white',name:'Cloud white',hex:'#e8e8e8'},{id:'black',name:'Midnight',hex:'#27272d'},{id:'blue',name:'Ocean blue',hex:'#306fc0'},{id:'red',name:'Cherry red',hex:'#c83f46'},{id:'green',name:'Sage green',hex:'#4f9254'},{id:'purple',name:'Violet',hex:'#99579b'}];

function GarmentExtractor({onExtracted}){
 const [file,setFile]=useState(null),[mode,setMode]=useState('garment'),[busy,setBusy]=useState(false),[message,setMessage]=useState(''),[preview,setPreview]=useState('');
 const [previewUrl,setPreviewUrl]=useState('');
 useEffect(()=>()=>{if(previewUrl)URL.revokeObjectURL(previewUrl);},[previewUrl]);
 const choose=e=>{const f=e.target.files?.[0];if(!f)return;if(previewUrl)URL.revokeObjectURL(previewUrl);setFile(f);setPreviewUrl(URL.createObjectURL(f));setPreview('');setMessage('');};
 const extract=async()=>{if(!file){setMessage('Choose a photo first.');return;}setBusy(true);setMessage(mode==='model'?'Detecting body landmarks and extracting the clothing region…':'Automatically separating the T-shirt from its background…');try{
   const body=new FormData();body.append('image',file);body.append('mode',mode);
   const res=await fetch(`${API}/api/wardrobe/extract-auto`,{method:'POST',body});const data=await res.json();if(!res.ok)throw new Error(data.detail||'Local extraction failed');
   const resultUrl=`${API}${data.preview_url}?t=${Date.now()}`;setPreview(resultUrl);setMessage(data.note||'Extraction complete. Inspect the transparent preview.');onExtracted(data.id,resultUrl);
 }catch(e){setMessage(e.message||'Local extraction failed.');}finally{setBusy(false);}};
 return <div className="extractor">
   <div className="section-kicker">T-SHIRT COLLECTION · LOCAL EXTRACTION</div>
   <b>Upload a T-shirt — automatic extraction</b>
   <p className="extract-help">Upload a real T-shirt photo. Best results: front-facing, fully visible, laid flat or on a hanger against a plain contrasting background. The same live body alignment is used, while the real garment's own cut and proportions are preserved.</p>
   <div className="extract-mode">
    <button type="button" className={mode==='garment'?'active':''} onClick={()=>{setMode('garment');setMessage('');}}>Clothing only <small>Flat lay / hanger</small></button>
    <button type="button" className={mode==='model'?'active':''} onClick={()=>{setMode('model');setMessage('');}}>On a model <small>Approximate cutout</small></button>
   </div>
   <label className="upload-label">Choose JPG, PNG or WebP<input type="file" accept="image/png,image/jpeg,image/webp" onChange={choose}/></label>
   {previewUrl&&<div className="extract-preview"><img src={previewUrl} alt="Selected source photo"/></div>}
   <button type="button" className="extract-run" onClick={extract} disabled={!file||busy}>{busy?'Extracting…':'Extract garment automatically'}</button>
   {message&&<div className="extract-message">{message}</div>}
   {preview&&<div className="extract-result"><b>Extracted garment preview</b><img src={preview} alt="Automatically extracted garment with transparent background"/><small>Review the edges before using this garment in the fitting room.</small></div>}
 </div>;
}

function App(){
 const videoRef=useRef(null),canvasRef=useRef(null),wsRef=useRef(null),streamRef=useRef(null),timerRef=useRef(null),busyRef=useRef(false),lastSend=useRef(0);
 const [camera,setCamera]=useState(false),[connecting,setConnecting]=useState(false),[connected,setConnected]=useState(false),[outfit,setOutfit]=useState('levis_black_tee'),[landmarks,setLandmarks]=useState(false),[error,setError]=useState(''),[fps,setFps]=useState(0),[frames,setFrames]=useState(0),[demoLoaded,setDemoLoaded]=useState(false),[demoLoading,setDemoLoading]=useState(false),[demoError,setDemoError]=useState(''),[extractedPreview,setExtractedPreview]=useState('');
 const demoInit=useRef(false);
 const loadDemo=useCallback(async()=>{setDemoLoading(true);setDemoError('');try{const response=await fetch(`${API}/api/wardrobe/demo`,{method:'POST'});const data=await response.json();if(!response.ok)throw new Error(data.detail||'Could not load demo garment');setDemoLoaded(true);setOutfit('levis_black_tee');}catch(e){setDemoError(e.message||'Could not load demo garment.');}finally{setDemoLoading(false);}},[]);
 useEffect(()=>{if(!demoInit.current){demoInit.current=true;loadDemo();}},[loadDemo]);
 const stop=useCallback(()=>{clearInterval(timerRef.current);timerRef.current=null;busyRef.current=false;if(wsRef.current){wsRef.current.close();wsRef.current=null;}if(streamRef.current){streamRef.current.getTracks().forEach(t=>t.stop());streamRef.current=null;}setCamera(false);setConnected(false);setConnecting(false);},[]);
 const connect=useCallback(async()=>{setError('');if(!demoLoaded){setError('The real T-shirt has not loaded yet. Wait for the demo image to finish loading.');return;}setConnecting(true);try{
   const stream=await navigator.mediaDevices.getUserMedia({video:{width:{ideal:960},height:{ideal:720},facingMode:'user'},audio:false});streamRef.current=stream;videoRef.current.srcObject=stream;await videoRef.current.play();setCamera(true);
   const ws=new WebSocket(WS);ws.binaryType='blob';wsRef.current=ws;
   ws.onopen=()=>{setConnected(true);setConnecting(false);ws.send(JSON.stringify({type:'settings',outfit,showLandmarks:landmarks}));startSending();};
   ws.onmessage=async ev=>{if(typeof ev.data==='string'){busyRef.current=false;try{const x=JSON.parse(ev.data);if(x.message)setError(x.message);}catch{}return;}try{const bmp=await createImageBitmap(ev.data);const c=canvasRef.current;if(c){c.width=bmp.width;c.height=bmp.height;c.getContext('2d').drawImage(bmp,0,0);bmp.close();setFrames(v=>v+1);}}finally{busyRef.current=false;}};
   ws.onerror=()=>setError('Cannot connect to the vision backend. Start the FastAPI server on port 8000.');
   ws.onclose=()=>{setConnected(false);setConnecting(false);};
  }catch(e){setConnecting(false);stop();setError(e.name==='NotAllowedError'?'Camera permission was denied. Allow camera access in your browser.':e.message||'Could not start camera.');}},[outfit,landmarks,stop,demoLoaded]);
 const startSending=()=>{clearInterval(timerRef.current);timerRef.current=setInterval(()=>{const v=videoRef.current,w=wsRef.current;if(!v||!w||w.readyState!==WebSocket.OPEN||v.readyState<2||busyRef.current)return;const now=performance.now();if(now-lastSend.current<50)return;lastSend.current=now;busyRef.current=true;const c=document.createElement('canvas');const scale=Math.min(1,400/v.videoWidth);c.width=Math.round(v.videoWidth*scale);c.height=Math.round(v.videoHeight*scale);c.getContext('2d').drawImage(v,0,0,c.width,c.height);c.toBlob(blob=>{if(blob&&w.readyState===WebSocket.OPEN)w.send(blob);else busyRef.current=false;},'image/jpeg',.60);},25);};
 useEffect(()=>()=>stop(),[stop]);
 useEffect(()=>{if(wsRef.current?.readyState===WebSocket.OPEN)wsRef.current.send(JSON.stringify({type:'settings',outfit,showLandmarks:landmarks}));},[outfit,landmarks]);
 const onExtracted=(id,preview)=>{setOutfit(id);setExtractedPreview(preview);setDemoLoaded(true);setDemoError('');};
 const reset=()=>{if(wsRef.current?.readyState===WebSocket.OPEN){wsRef.current.send(JSON.stringify({type:'settings',outfit,showLandmarks:false}));setLandmarks(false);}setFrames(0);};
 return <div className="app-shell"><header className="topbar"><a className="brand" href="#"><span className="brand-mark"><Shirt size={20}/></span><span>AR<span className="brand-accent">WARDROBE</span><small>VIRTUAL FITTING STUDIO</small></span></a><div className="top-actions"><span className="phase-pill"><span/> T-SHIRT TRY-ON</span><span className="api-label"><span className={connected?'dot live':'dot'}/>{connected?'Vision engine connected':'Vision engine offline'}</span></div></header>
 <main className="main"><section className="intro"><div className="eyebrow"><Sparkles size={14}/> YOUR PERSONAL VIRTUAL FITTING ROOM</div><h1>Try it on.<br/><em>See it move.</em></h1><p>Choose a T-shirt or extract one from a catalogue photo. Your camera feed is processed locally by the vision backend.</p></section>
 <section className="workspace"><div className="studio-card"><div className="studio-head"><div><div className="section-kicker">LIVE STUDIO</div><h2>Fitting room</h2></div><div className={'status '+(connected?'status-live':'') }><span className="dot"/>{connected?'LIVE':'STANDBY'}</div></div>
 <div className="preview"><video ref={videoRef} className="hidden-video" playsInline muted/><canvas ref={canvasRef}/>{!camera&&<div className="empty-state"><div className="empty-icon"><Camera size={30}/></div><h3>Your fitting room is ready</h3><p>Start your camera to see the virtual try-on experience.</p><button className="primary-btn" onClick={connect} disabled={connecting||!demoLoaded}>{connecting?'Connecting…':<><Camera size={17}/> Start camera</>}</button></div>}{camera&&<div className="preview-tag"><Scan size={14}/> BODY TRACKING</div>}</div>
 <div className="studio-footer"><div className="metric"><Activity size={16}/><span>{connected?'Processing frames':'Awaiting camera'}</span></div><div className="metric-value">{frames?`${frames} frames`:'—'}</div><div className="studio-controls">{camera?<><button className="icon-btn" title="Reset view" onClick={reset}><RefreshCw size={17}/></button><button className="stop-btn" onClick={stop}><CameraOff size={16}/> Stop camera</button></>:<button className="secondary-btn" onClick={connect} disabled={connecting||!demoLoaded}>{connecting?'Connecting…':'Connect camera'}</button>}</div></div>
 {error&&<div className="error-banner"><AlertTriangle size={16}/>{error}</div>}</div>
 <aside className="side-panel"><div className="panel-card"><div className="section-kicker">THE COLLECTION</div><div className="panel-title"><h2>Choose or extract a T-shirt</h2><span className="count">06</span></div><p className="muted">Start with the bundled test shirt, or upload a T-shirt for automatic local extraction. No manual outline is needed.</p><div className="outfit-grid">{outfits.map(o=><button key={o.id} onClick={()=>setOutfit(o.id)} className={'outfit '+(outfit===o.id?'selected':'')}><span className="swatch" style={{'--swatch':o.hex}}>{outfit===o.id&&<Check size={17}/>}</span><span>{o.name}</span></button>)}</div><div className="demo-garment"><div className="demo-garment-image"><img src={extractedPreview||`${API}/api/wardrobe/demo-preview`} alt="Current isolated test or extracted garment" onError={e=>{e.currentTarget.style.opacity='.25';}}/></div><div className="demo-garment-copy"><span className="demo-label">BUNDLED TEST GARMENT</span><b>Clean test T-shirt</b><small>{demoLoaded?'Safe isolated illustration loaded':'Bundled clean test asset'}</small><small>{outfit==="custom_upload"?"Your extracted garment is selected":"Use the bundled illustration to verify tracking"}</small></div><button className="load-demo-btn" onClick={loadDemo} disabled={demoLoading}>{demoLoading?'Loading…':'Use test tee'}</button></div>{demoError&&<div className="demo-error">{demoError}</div>}<p className="demo-disclaimer">Bundled clean test garment. Upload your own product photo below to extract a shirt.</p><GarmentExtractor onExtracted={onExtracted}/></div>
 <div className="panel-card settings-card"><div className="section-kicker">TRACKING SETTINGS</div><h2>Studio controls</h2><label className="toggle-row"><span><Scan size={17}/><span><b>Show body landmarks</b><small>Debug full-body pose landmarks (up to 33 points)</small></span></span><input type="checkbox" checked={landmarks} onChange={e=>setLandmarks(e.target.checked)}/></label><div className="setting-note"><Activity size={16}/><span>Adaptive smoothing is enabled in the backend to reduce tracking jitter.</span></div></div>
 <div className="info-card"><div className="info-icon"><CircleHelp size={18}/></div><div><b>Getting the best result</b><p>Face the camera, keep shoulders and hips visible, use even lighting and stand far enough back for your upper body to fit in frame.</p></div></div></aside></section>
 <section className="bottom-strip"><div><span className="strip-number">01</span><span><b>Pose detection</b><small>MediaPipe landmarks</small></span></div><div><span className="strip-number">02</span><span><b>Stable tracking</b><small>One Euro smoothing</small></span></div><div><span className="strip-number">03</span><span><b>Garment overlay</b><small>Piecewise-affine garment mesh</small></span></div><div><span className="strip-number">04</span><span><b>Live rendering</b><small>FastAPI + WebSocket</small></span></div></section>
 <footer>AR WARDROBE <span>•</span> T-SHIRT VIRTUAL TRY-ON <span>•</span> LOCAL CAMERA PROCESSING</footer></main></div>
}

createRoot(document.getElementById('root')).render(<React.StrictMode><App/></React.StrictMode>);
