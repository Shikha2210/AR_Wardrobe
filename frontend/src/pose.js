// Loads MediaPipe Pose Landmarker (self-hosted model + WASM, no CDN dependency).
import { FilesetResolver, PoseLandmarker } from '@mediapipe/tasks-vision';

const base = () => new URL(`${import.meta.env.BASE_URL}mp/`, window.location.href).href;
let fileset;
const files = () => (fileset ||= FilesetResolver.forVisionTasks(base() + 'wasm'));

async function make(mode, delegate) {
  return PoseLandmarker.createFromOptions(await files(), {
    baseOptions: { modelAssetPath: base() + 'pose_landmarker_lite.task', delegate },
    runningMode: mode, numPoses: 1, outputSegmentationMasks: true,
    minPoseDetectionConfidence: 0.55, minPosePresenceConfidence: 0.55, minTrackingConfidence: 0.55,
  });
}

let video, image;
export function loadVideoPose() {
  return (video ||= make('VIDEO', 'GPU').catch(() => make('VIDEO', 'CPU')));
}
export function loadImagePose() {
  return (image ||= make('IMAGE', 'CPU'));
}
