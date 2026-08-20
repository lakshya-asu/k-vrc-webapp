import React from 'react';
import {
  AbsoluteFill,
  Easing,
  interpolate,
  useCurrentFrame,
  useVideoConfig,
} from 'remotion';
import {FaceCanvas, FaceJob} from './FaceCanvas';
import {LED_WARM} from './led';

// End card: K-VRC's neutral face, then the honest credit line.
const END_JOB: FaceJob = {
  fps: 30,
  frame_end: 150,
  face_beats: [
    {at_ms: 0, duration_ms: 1200, expression: 'neutral_idle', intensity: 1},
  ],
  viseme_samples: [],
  seed: 7,
};

export const EndCard: React.FC = () => {
  const frame = useCurrentFrame();
  const {height, durationInFrames} = useVideoConfig();
  const fadeIn = interpolate(frame, [0, 16], [0, 1], {
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
    easing: Easing.inOut(Easing.cubic),
  });
  const textIn = interpolate(frame, [20, 38], [0, 1], {
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
    easing: Easing.inOut(Easing.quad),
  });
  const fadeOut = interpolate(
    frame,
    [durationInFrames - 18, durationInFrames - 1],
    [1, 0],
    {
      extrapolateLeft: 'clamp',
      extrapolateRight: 'clamp',
      easing: Easing.inOut(Easing.quad),
    },
  );
  const mono = '"Consolas", "Courier New", monospace';
  return (
    <AbsoluteFill
      style={{
        backgroundColor: '#000',
        alignItems: 'center',
        justifyContent: 'center',
        flexDirection: 'column',
        gap: 36,
        opacity: fadeOut,
      }}
    >
      <div style={{opacity: fadeIn}}>
        <FaceCanvas
          job={END_JOB}
          style={{width: height * 0.42, height: height * 0.42}}
        />
      </div>
      <div
        style={{
          opacity: textIn,
          textAlign: 'center',
          fontFamily: mono,
          color: LED_WARM.primary,
          textShadow: `0 0 18px ${LED_WARM.primary}`,
        }}
      >
        <div style={{fontSize: 46, letterSpacing: 6}}>
          performed live by a local model
        </div>
        <div
          style={{
            fontSize: 28,
            letterSpacing: 12,
            marginTop: 26,
            opacity: 0.75,
          }}
        >
          PROJECT ANIMUS
        </div>
      </div>
    </AbsoluteFill>
  );
};
