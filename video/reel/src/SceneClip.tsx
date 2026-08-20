import React from 'react';
import {
  AbsoluteFill,
  OffthreadVideo,
  interpolate,
  staticFile,
  useCurrentFrame,
} from 'remotion';
import {LED_WARM} from './led';

export type SceneMeta = {
  num: number;
  name: string;
  description: string;
  file: string;
  durationS: number;
};

const mono = '"Consolas", "Courier New", monospace';

// Lower third: scene number + name, one-line direction. LED mono look.
const LowerThird: React.FC<{scene: SceneMeta; frames: number}> = ({
  scene,
  frames,
}) => {
  const frame = useCurrentFrame();
  const slide = interpolate(frame, [10, 24], [40, 0], {
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
  });
  const alpha =
    interpolate(frame, [10, 24], [0, 1], {
      extrapolateLeft: 'clamp',
      extrapolateRight: 'clamp',
    }) *
    interpolate(frame, [frames - 22, frames - 10], [1, 0], {
      extrapolateLeft: 'clamp',
      extrapolateRight: 'clamp',
    });
  return (
    <div
      style={{
        position: 'absolute',
        left: 72,
        bottom: 56 - slide * -1,
        transform: `translateY(${slide}px)`,
        opacity: alpha,
        fontFamily: mono,
        color: LED_WARM.primary,
        textShadow: `0 0 12px ${LED_WARM.primary}`,
        background: 'rgba(0, 0, 0, 0.55)',
        borderLeft: `6px solid ${LED_WARM.primary}`,
        padding: '14px 26px 16px 20px',
        maxWidth: 900,
      }}
    >
      <div style={{fontSize: 30, letterSpacing: 5}}>
        {String(scene.num).padStart(2, '0')} · {scene.name.toUpperCase()}
      </div>
      <div style={{fontSize: 22, opacity: 0.85, marginTop: 6}}>
        {scene.description}
      </div>
    </div>
  );
};

// One scene: the rendered take full-frame with a short fade at both
// ends (the fades overlap neighbors in the series, making crossfades)
// plus its lower-third caption.
export const SceneClip: React.FC<{
  scene: SceneMeta;
  frames: number;
  cross: number;
}> = ({scene, frames, cross}) => {
  const frame = useCurrentFrame();
  const fadeIn = interpolate(frame, [0, cross], [0, 1], {
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
  });
  const fadeOut = interpolate(frame, [frames - cross, frames - 1], [1, 0], {
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
  });
  return (
    <AbsoluteFill style={{backgroundColor: '#000'}}>
      <AbsoluteFill style={{opacity: fadeIn * fadeOut}}>
        <OffthreadVideo
          src={staticFile(scene.file)}
          style={{width: '100%', height: '100%', objectFit: 'cover'}}
        />
        <LowerThird scene={scene} frames={frames} />
      </AbsoluteFill>
    </AbsoluteFill>
  );
};
