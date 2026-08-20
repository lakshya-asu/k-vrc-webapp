import React from 'react';
import {
  AbsoluteFill,
  Easing,
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
  const slide = interpolate(frame, [10, 26], [40, 0], {
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
    easing: Easing.out(Easing.cubic),
  });
  const alpha =
    interpolate(frame, [10, 26], [0, 1], {
      extrapolateLeft: 'clamp',
      extrapolateRight: 'clamp',
      easing: Easing.inOut(Easing.quad),
    }) *
    interpolate(frame, [frames - 24, frames - 12], [1, 0], {
      extrapolateLeft: 'clamp',
      extrapolateRight: 'clamp',
      easing: Easing.inOut(Easing.quad),
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

// One scene: the rendered take full-frame plus its lower-third caption.
//
// Transitions (reel-polish brief, directive 4): the clip's first and
// last `cross` frames overlap its neighbors in the series. Video
// opacity eases in and out (cubic in-out, no linear pops); audio runs
// an equal-power crossfade over exactly the same frames, so at every
// point of a join sin^2 + cos^2 keeps the summed energy constant and
// the waveform carries no hard edge.
export const SceneClip: React.FC<{
  scene: SceneMeta;
  frames: number;
  cross: number;
}> = ({scene, frames, cross}) => {
  const frame = useCurrentFrame();
  const fadeProgress = (f: number) =>
    interpolate(f, [0, cross], [0, 1], {
      extrapolateLeft: 'clamp',
      extrapolateRight: 'clamp',
    });
  const inP = fadeProgress(frame);
  const outP = fadeProgress(frames - 1 - frame);
  const easeInOut = (p: number) =>
    interpolate(p, [0, 1], [0, 1], {easing: Easing.inOut(Easing.cubic)});
  // Equal-power gains for the same windows the opacity uses.
  const audioGain = (f: number) =>
    Math.sin((fadeProgress(f) * Math.PI) / 2) *
    Math.sin((fadeProgress(frames - 1 - f) * Math.PI) / 2);
  return (
    <AbsoluteFill style={{backgroundColor: '#000'}}>
      <AbsoluteFill style={{opacity: easeInOut(inP) * easeInOut(outP)}}>
        <OffthreadVideo
          src={staticFile(scene.file)}
          volume={(f) => audioGain(f)}
          style={{width: '100%', height: '100%', objectFit: 'cover'}}
        />
        <LowerThird scene={scene} frames={frames} />
      </AbsoluteFill>
    </AbsoluteFill>
  );
};
