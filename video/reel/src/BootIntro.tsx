import React from 'react';
import {
  AbsoluteFill,
  Easing,
  interpolate,
  useCurrentFrame,
  useVideoConfig,
} from 'remotion';
import {FaceCanvas, FaceJob} from './FaceCanvas';

// Cold open: the K-VRC visor boots full-frame, exactly the face the
// character wears in the scenes (boot_init expression from the shared
// expression library), with a forced LED glitch as it wakes.
const BOOT_JOB: FaceJob = {
  fps: 30,
  frame_end: 90,
  face_beats: [
    {at_ms: 0, duration_ms: 900, expression: 'boot_init', intensity: 1},
    {at_ms: 1700, duration_ms: 800, expression: 'neutral_idle', intensity: 1},
  ],
  viseme_samples: [],
  force_glitches: [
    {at_ms: 250, duration_ms: 180},
    {at_ms: 1450, duration_ms: 140},
  ],
  seed: 42,
};

export const BootIntro: React.FC = () => {
  const frame = useCurrentFrame();
  const {height, durationInFrames} = useVideoConfig();
  const fadeIn = interpolate(frame, [0, 10], [0, 1], {
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
    easing: Easing.inOut(Easing.quad),
  });
  const fadeOut = interpolate(
    frame,
    [durationInFrames - 12, durationInFrames - 1],
    [1, 0],
    {
      extrapolateLeft: 'clamp',
      extrapolateRight: 'clamp',
      easing: Easing.inOut(Easing.quad),
    },
  );
  const size = height * interpolate(frame, [0, 74], [0.96, 1.04]);
  return (
    <AbsoluteFill
      style={{
        backgroundColor: '#000',
        alignItems: 'center',
        justifyContent: 'center',
      }}
    >
      <div style={{opacity: fadeIn * fadeOut}}>
        <FaceCanvas job={BOOT_JOB} style={{width: size, height: size}} />
      </div>
    </AbsoluteFill>
  );
};
