import React, {useEffect, useMemo, useRef} from 'react';
import {useCurrentFrame} from 'remotion';
// The actual visor face renderer, shared verbatim with the webapp and
// the take pipeline (src/animus/face/). It draws on any 2D canvas.
// @ts-ignore
import {EXPRESSION_LIBRARY} from '../../../src/animus/face/expressionLibrary.js';
// @ts-ignore
import {drawFaceFrame, W, H} from '../../../src/animus/face/faceScreenDraw.js';
// @ts-ignore
import {buildFaceTimeline, mulberry32} from '../../../src/animus/face/faceTimeline.js';

export type FaceJob = {
  fps: number;
  frame_end: number;
  face_beats: {
    at_ms: number;
    duration_ms: number;
    expression: string;
    intensity?: number;
  }[];
  viseme_samples: unknown[];
  force_glitches?: {at_ms: number; duration_ms: number}[];
  seed: number;
};

// Renders the K-VRC LED face for the current Remotion frame by driving
// the ported webapp canvas renderer with a deterministic timeline.
export const FaceCanvas: React.FC<{
  job: FaceJob;
  style?: React.CSSProperties;
}> = ({job, style}) => {
  const frame = useCurrentFrame();
  const ref = useRef<HTMLCanvasElement>(null);
  const timeline = useMemo(
    () => buildFaceTimeline(job, EXPRESSION_LIBRARY),
    [job],
  );

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    const state = timeline[Math.min(frame, timeline.length - 1)];
    // One rng stream per frame keeps renders deterministic under
    // Remotion's parallel frame rendering.
    const rng = mulberry32(((job.seed >>> 0) * 100003 + frame) >>> 0);
    drawFaceFrame(ctx, state, rng);
  }, [frame, timeline, job.seed]);

  return (
    <canvas
      ref={ref}
      width={W}
      height={H}
      style={{imageRendering: 'pixelated', ...style}}
    />
  );
};
