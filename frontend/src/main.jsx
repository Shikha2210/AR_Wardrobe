import React, {useEffect, useRef, useState, useCallback} from 'react';
import {createRoot} from 'react-dom/client';
import {Camera, CameraOff, Shirt, RefreshCw, Activity, Wifi, WifiOff, Scan, Sparkles, CircleHelp, Check, AlertTriangle} from 'lucide-react';
import {PoseLandmarker} from '@mediapipe/tasks-vision';
import './style.css';
import {loadVideoPose} from './pose.js';
import {LandmarkTracker} from './tracking.js';
import {TryOnEngine,DEMO_ID} from './tryon.js';
import {extractGarment} from './extract.js';

const BASE=import.meta.env.BASE_URL;
const PROC_W=480;

const outfits=[{id:'white',name:'Cloud white',hex:'#e8e8e8'},{id:'black',name:'Midnight',hex:'#27272d'},{id:'blue',name:'Ocean blue',hex:'#306fc0'},{id:'red',name:'Cherry red',hex:'#c83f46'},{id:'green',name:'Sage green',hex:'#4f9254'},{id:'purple',name:'Violet',hex:'#99579b'}];

function GarmentExtractor({onExtracted}){
 const [file,setFile]=useState(null),[mode,setMode]=useState('garment'),[busy,setBusy]=useState(false),[message,setMessage]=useState(''),[preview,setPreview]=useState('');
 const [previewUrl,setPreviewUrl]=useState('');
 useEffect(()=>()=>{if(previewUrl)URL.revokeObjectURL(previewUrl);},[previewUrl]);
 const choose=e=>{const f=e.target.files?.[0];if(!f)return;if(previewUrl)URL.revokeObjectURL(previewUrl);setFile(f);setPreviewUrl(URL.createObjectURL(f));setPreview('');setMessage('');};
 const extract=async()=>{if(!file){setMessage('Choose a photo first.');return;}setBusy(true);setMessage(mode==='model'?'Detecting body landmarks and extracting the clothing region…':'Automatically separating the T-shirt from its background…');try{
   const canvas=await extractGarment(file,mode);
   const url=canvas.toDataURL('image/png');setPreview(url);setMessage((mode==='model'?'MediaPipe Pose plus person segmentation created an approximate upper-body cutout. Arms or hidden fabric may remain imperfect.':'Background removed automatically in your browser; no manual points used.')+' Inspect the preview before using it.');onExtracted('custom_upload',url,canvas);
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
 const videoRef=useRef(null),canvasRef=useRef(null),streamRef=useRef(null),rafRef=useRef(0);
 const poseRef=useRef(null),engineRef=useRef(new TryOnEngine()),trackerRef=useRef(new LandmarkTracker());
 const procRef=useRef(null),maskRef=useRef(null),maskImg=useRef(null),outfitRef=useRef(DEMO_ID),landmarksRef=useRef(false),statsRef=useRef({n:0,last:0,fps:0});
 const [camera,setCamera]=useState(false),[connecting,setConnecting]=useState(false),[ready,setReady]=useState(false),[outfit,setOutfit]=useState(DEMO_ID),[landmarks,setLandmarks]=useState(false),[error,setError]=useState(''),[frames,setFrames]=useState(0),[demoLoaded,setDemoLoaded]=useState(false),[demoError,setDemoError]=useState(''),[extractedPreview,setExtractedPreview]=useState('');
 useEffect(()=>{outfitRef.current=outfit;},[outfit]);
 useEffect(()=>{landmarksRef.current=landmarks;},[landmarks]);
 const loadDemo=useCallback(()=>{setDemoError('');return new Promise(res=>{const img=new Image();img.onload=()=>{engineRef.current.loadBase(img);setDemoLoaded(true);setOutfit(DEMO_ID);res();};img.onerror=()=>{setDemoError('Could not load the bundled test T-shirt.');res();};img.src=`${BASE}tshirt.png`;});},[]);
 const demoInit=useRef(false);
 useEffect(()=>{if(!demoInit.current){demoInit.current=true;loadDemo();}},[loadDemo]);
 const ensurePose=useCallback(async()=>{if(poseRef.current)return poseRef.current;poseRef.current=await loadVideoPose();setReady(true);return poseRef.current;},[]);
 const stop=useCallback(()=>{cancelAnimationFrame(rafRef.current);rafRef.current=0;if(streamRef.current){streamRef.current.getTracks().forEach(t=>t.stop());streamRef.current=null;}trackerRef.current.reset();setCamera(false);setConnecting(false);},[]);
 const tick=useCallback(()=>{
  rafRef.current=requestAnimationFrame(tick);
  const v=videoRef.current,pose=poseRef.current,out=canvasRef.current;if(!v||!pose||!out||v.readyState<2||!v.videoWidth)return;
  const now=performance.now(),st=statsRef.current;if(now-st.last<28)return;
  const inst=1000/Math.max(now-st.last,1);st.fps=st.fps?0.2*inst+0.8*st.fps:inst;st.last=now;
  const pw=PROC_W,ph=Math.round(v.videoHeight*pw/v.videoWidth);
  if(!procRef.current)procRef.current=document.createElement('canvas');
  const proc=procRef.current;if(proc.width!==pw||proc.height!==ph){proc.width=pw;proc.height=ph;}
  const pg=proc.getContext('2d');pg.save();pg.translate(pw,0);pg.scale(-1,1);pg.drawImage(v,0,0,pw,ph);pg.restore();
  let res;try{res=pose.detectForVideo(proc,now);}catch(e){setError('Pose detection failed: '+(e.message||e));return;}
  const lms=res.landmarks&&res.landmarks[0];
  const {pts,status}=trackerRef.current.update(lms,pw,ph,now/1000);
  if(out.width!==pw||out.height!==ph){out.width=pw;out.height=ph;}
  const ctx=out.getContext('2d');ctx.drawImage(proc,0,0);
  const seg=res.segmentationMasks&&res.segmentationMasks[0];
  if(pts){
   let mask=null;
   if(seg){
    const f=seg.getAsFloat32Array();
    if(!maskRef.current)maskRef.current=document.createElement('canvas');
    const mc=maskRef.current;if(mc.width!==pw||mc.height!==ph){mc.width=pw;mc.height=ph;maskImg.current=null;}
    const mg=mc.getContext('2d');if(!maskImg.current)maskImg.current=mg.createImageData(pw,ph);
    const d=maskImg.current.data;let cover=0;
    // Sanity check: the torso centre must be "person" in the mask, otherwise skip clipping this frame.
    const cx=Math.round((pts[11][0]+pts[12][0]+pts[23][0]+pts[24][0])/4),cy=Math.round((pts[11][1]+pts[12][1]+pts[23][1]+pts[24][1])/4);
    if(cx>=0&&cy>=0&&cx<pw&&cy<ph)cover=f[cy*pw+cx];
    if(cover>0.4){for(let i=0;i<f.length;i++){d[i*4]=255;d[i*4+1]=255;d[i*4+2]=255;d[i*4+3]=Math.min(255,f[i]*1.5*255);}mg.putImageData(maskImg.current,0,0);mask=mc;}
   }
   engineRef.current.render(ctx,proc,pts,outfitRef.current,mask);
   if(landmarksRef.current){ctx.strokeStyle='rgb(70,220,110)';ctx.lineWidth=2;for(const c of PoseLandmarker.POSE_CONNECTIONS){if(pts[c.start]&&pts[c.end]){ctx.beginPath();ctx.moveTo(pts[c.start][0],pts[c.start][1]);ctx.lineTo(pts[c.end][0],pts[c.end][1]);ctx.stroke();}}ctx.fillStyle='rgb(40,240,100)';for(const k of Object.keys(pts)){ctx.beginPath();ctx.arc(pts[k][0],pts[k][1],[11,12,13,14,15,16,23,24].includes(+k)?4:2,0,7);ctx.fill();}}
  }
  if(res.segmentationMasks)res.segmentationMasks.forEach(m=>m.close&&m.close());
  ctx.font='600 13px sans-serif';ctx.fillStyle='rgb(80,255,130)';ctx.fillText(`${status} | ${Math.round(st.fps)} FPS`,12,24);
  st.n++;if(st.n%10===0)setFrames(st.n);
 },[]);
 const connect=useCallback(async()=>{setError('');if(!demoLoaded){setError('The T-shirt has not loaded yet. Wait a moment and try again.');return;}setConnecting(true);try{
   await ensurePose();
   const stream=await navigator.mediaDevices.getUserMedia({video:{width:{ideal:960},height:{ideal:720},facingMode:'user'},audio:false});streamRef.current=stream;videoRef.current.srcObject=stream;await videoRef.current.play();
   statsRef.current={n:0,last:0,fps:0};setFrames(0);setCamera(true);setConnecting(false);cancelAnimationFrame(rafRef.current);rafRef.current=requestAnimationFrame(tick);
  }catch(e){stop();setError(e.name==='NotAllowedError'?'Camera permission was denied. Allow camera access in your browser.':(e.message||'Could not start camera or load the vision model.'));}},[demoLoaded,ensurePose,stop,tick]);
 useEffect(()=>()=>stop(),[stop]);
 const onExtracted=(id,preview,canvas)=>{engineRef.current.setCustom(canvas);setOutfit(id);setExtractedPreview(preview);setDemoLoaded(true);setDemoError('');};
 const reset=()=>{setLandmarks(false);trackerRef.current.reset();statsRef.current.n=0;setFrames(0);};
 const connected=camera&&ready;
 return <div className="app-shell"><header className="topbar"><a className="brand" href="#"><span className="brand-mark"><Shirt size={20}/></span><span>AR<span className="brand-accent">WARDROBE</span><small>VIRTUAL FITTING STUDIO</small></span></a><div className="top-actions"><span className="phase-pill"><span/> T-SHIRT TRY-ON</span><span className="api-label"><span className={connected?'dot live':'dot'}/>{connected?'Vision engine running in your browser':(ready?'Vision engine ready':'Vision engine loads on start')}</span></div></header>
 <main className="main"><section className="intro"><div className="eyebrow"><Sparkles size={14}/> YOUR PERSONAL VIRTUAL FITTING ROOM</div><h1>Try it on.<br/><em>See it move.</em></h1><p>Choose a T-shirt or extract one from a catalogue photo. Your camera feed never leaves your device; everything runs locally in your browser.</p></section>
 <section className="workspace"><div className="studio-card"><div className="studio-head"><div><div className="section-kicker">LIVE STUDIO</div><h2>Fitting room</h2></div><div className={'status '+(connected?'status-live':'') }><span className="dot"/>{connected?'LIVE':'STANDBY'}</div></div>
 <div className="preview"><video ref={videoRef} className="hidden-video" playsInline muted/><canvas ref={canvasRef}/>{!camera&&<div className="empty-state"><div className="empty-icon"><Camera size={30}/></div><h3>Your fitting room is ready</h3><p>Start your camera to see the virtual try-on experience.</p><button className="primary-btn" onClick={connect} disabled={connecting||!demoLoaded}>{connecting?'Loading vision model…':<><Camera size={17}/> Start camera</>}</button></div>}{camera&&<div className="preview-tag"><Scan size={14}/> BODY TRACKING</div>}</div>
 <div className="studio-footer"><div className="metric"><Activity size={16}/><span>{connected?'Processing frames':'Awaiting camera'}</span></div><div className="metric-value">{frames?`${frames} frames`:'—'}</div><div className="studio-controls">{camera?<><button className="icon-btn" title="Reset view" onClick={reset}><RefreshCw size={17}/></button><button className="stop-btn" onClick={stop}><CameraOff size={16}/> Stop camera</button></>:<button className="secondary-btn" onClick={connect} disabled={connecting||!demoLoaded}>{connecting?'Loading vision model…':'Connect camera'}</button>}</div></div>
 {error&&<div className="error-banner"><AlertTriangle size={16}/>{error}</div>}</div>
 <aside className="side-panel"><div className="panel-card"><div className="section-kicker">THE COLLECTION</div><div className="panel-title"><h2>Choose or extract a T-shirt</h2><span className="count">06</span></div><p className="muted">Start with the bundled test shirt, or upload a T-shirt for automatic local extraction. No manual outline is needed.</p><div className="outfit-grid">{outfits.map(o=><button key={o.id} onClick={()=>setOutfit(o.id)} className={'outfit '+(outfit===o.id?'selected':'')}><span className="swatch" style={{'--swatch':o.hex}}>{outfit===o.id&&<Check size={17}/>}</span><span>{o.name}</span></button>)}</div><div className="demo-garment"><div className="demo-garment-image"><img src={extractedPreview||`${BASE}tshirt.png`} alt="Current isolated test or extracted garment" onError={e=>{e.currentTarget.style.opacity='.25';}}/></div><div className="demo-garment-copy"><span className="demo-label">BUNDLED TEST GARMENT</span><b>Clean test T-shirt</b><small>{demoLoaded?'Safe isolated illustration loaded':'Bundled clean test asset'}</small><small>{outfit==="custom_upload"?"Your extracted garment is selected":"Use the bundled illustration to verify tracking"}</small></div><button className="load-demo-btn" onClick={()=>{setOutfit(DEMO_ID);setExtractedPreview('');}}>Use test tee</button></div>{demoError&&<div className="demo-error">{demoError}</div>}<p className="demo-disclaimer">Bundled clean test garment. Upload your own product photo below to extract a shirt.</p><GarmentExtractor onExtracted={onExtracted}/></div>
 <div className="panel-card settings-card"><div className="section-kicker">TRACKING SETTINGS</div><h2>Studio controls</h2><label className="toggle-row"><span><Scan size={17}/><span><b>Show body landmarks</b><small>Debug full-body pose landmarks (up to 33 points)</small></span></span><input type="checkbox" checked={landmarks} onChange={e=>setLandmarks(e.target.checked)}/></label><div className="setting-note"><Activity size={16}/><span>Adaptive smoothing is enabled to reduce tracking jitter.</span></div></div>
 <div className="info-card"><div className="info-icon"><CircleHelp size={18}/></div><div><b>Getting the best result</b><p>Face the camera, keep shoulders and hips visible, use even lighting and stand far enough back for your upper body to fit in frame.</p></div></div></aside></section>
 <section className="bottom-strip"><div><span className="strip-number">01</span><span><b>Pose detection</b><small>MediaPipe landmarks</small></span></div><div><span className="strip-number">02</span><span><b>Stable tracking</b><small>One Euro smoothing</small></span></div><div><span className="strip-number">03</span><span><b>Garment overlay</b><small>Piecewise-affine garment mesh</small></span></div><div><span className="strip-number">04</span><span><b>Live rendering</b><small>HTML5 canvas, in-browser</small></span></div></section>
 <footer>AR WARDROBE <span>•</span> T-SHIRT VIRTUAL TRY-ON <span>•</span> LOCAL CAMERA PROCESSING</footer></main></div>
}

createRoot(document.getElementById('root')).render(<React.StrictMode><App/></React.StrictMode>);
