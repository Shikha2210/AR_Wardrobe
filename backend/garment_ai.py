"""Local automatic CV garment extraction. Not semantic clothing AI.
Uses existing OpenCV + MediaPipe only; avoids blocked native model stack.
"""
from io import BytesIO
import cv2, numpy as np
from PIL import Image

def _decode(raw):
    try: rgb=np.asarray(Image.open(BytesIO(raw)).convert("RGB"),dtype=np.uint8)
    except Exception as e: raise ValueError("Could not open uploaded image.") from e
    h,w=rgb.shape[:2]
    if min(h,w)<80: raise ValueError("Image must be at least 80 x 80 pixels.")
    if max(h,w)>1600:
        q=1600/max(h,w); rgb=cv2.resize(rgb,(int(w*q),int(h*q)),interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)

def _grab(image, rect, polygon=None):
    h,w=image.shape[:2]; x,y,rw,rh=map(int,rect)
    x=max(1,min(x,w-2)); y=max(1,min(y,h-2)); rw=max(2,min(rw,w-x-1)); rh=max(2,min(rh,h-y-1))
    mask=np.zeros((h,w),np.uint8); bg=np.zeros((1,65),np.float64); fg=np.zeros((1,65),np.float64)
    if polygon is None: cv2.grabCut(image,mask,(x,y,rw,rh),bg,fg,5,cv2.GC_INIT_WITH_RECT)
    else:
        cv2.fillPoly(mask,[np.asarray(polygon,np.int32)],cv2.GC_PR_FGD)
        k=max(3,(min(h,w)//35)|1); core=cv2.erode((mask==cv2.GC_PR_FGD).astype(np.uint8),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(k,k)))
        mask[core>0]=cv2.GC_FGD; cv2.grabCut(image,mask,None,bg,fg,5,cv2.GC_INIT_WITH_MASK)
    alpha=np.where((mask==cv2.GC_FGD)|(mask==cv2.GC_PR_FGD),255,0).astype(np.uint8)
    n,lab,stats,_=cv2.connectedComponentsWithStats((alpha>0).astype(np.uint8),8)
    if n>1: alpha=np.where(lab==1+int(np.argmax(stats[1:,cv2.CC_STAT_AREA])),alpha,0).astype(np.uint8)
    alpha=cv2.GaussianBlur(cv2.morphologyEx(alpha,cv2.MORPH_CLOSE,np.ones((5,5),np.uint8)),(3,3),.7)
    rgba=cv2.cvtColor(image,cv2.COLOR_BGR2RGBA); rgba[:,:,3]=alpha; ys,xs=np.where(alpha>12)
    if len(xs)<500: raise ValueError("No clear garment found. Use a centred garment and uncluttered background.")
    return rgba[ys.min():ys.max()+1,xs.min():xs.max()+1]

def _model(image):
    h,w=image.shape[:2]
    import mediapipe as mp
    with mp.solutions.pose.Pose(static_image_mode=True,model_complexity=0,enable_segmentation=False,min_detection_confidence=.5) as pose: result=pose.process(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
    if not result.pose_landmarks: raise ValueError("No person detected. Use a front-facing upper-body photo.")
    lm=result.pose_landmarks.landmark
    if min(lm[i].visibility for i in (11,12,23,24))<.45: raise ValueError("Shoulders and hips must be visible.")
    def pt(i): return np.array([int(lm[i].x*w),int(lm[i].y*h)],np.int32)
    ls,rs,le,re,lh,rh=[pt(i) for i in (11,12,13,14,23,24)]; sw=np.linalg.norm(ls-rs)
    if sw<25 or np.linalg.norm((lh+rh)/2-(ls+rs)/2)<35: raise ValueError("Person is too small; upload a closer photo.")
    poly=np.array([ls+[-int(.1*sw),-int(.02*sw)],rs+[int(.1*sw),-int(.02*sw)],re+(rs-re)*.1,rh+[int(.06*sw),int(.03*np.linalg.norm(rh-ls))],lh+[-int(.06*sw),int(.03*np.linalg.norm(lh-rs))],le+(ls-le)*.1],np.int32)
    x0=max(1,int(poly[:,0].min())); y0=max(1,int(poly[:,1].min())); x1=min(w-1,int(poly[:,0].max())); y1=min(h-1,int(poly[:,1].max()))
    if x1-x0<30 or y1-y0<30: raise ValueError("Detected garment region too small.")
    return _grab(image,(x0,y0,x1-x0,y1-y0),poly)

def extract_garment_rgba(raw,mode):
    image=_decode(raw); h,w=image.shape[:2]
    if mode=="garment": return _grab(image,(int(.02*w),int(.01*h),int(.96*w),int(.98*h))),"Local OpenCV GrabCut automatically separated the centred garment; no manual points used."
    if mode=="model": return _model(image),"MediaPipe Pose plus OpenCV GrabCut created an automatic approximate upper-body cutout. Arms or hidden fabric may remain imperfect."
    raise ValueError("Unknown extraction mode. Choose garment or model.")
