import React from 'react';
import {AbsoluteFill, Series} from 'remotion';
import {BootIntro} from './BootIntro';
import {EndCard} from './EndCard';
import {SceneClip, SceneMeta} from './SceneClip';
import {TitleCard} from './TitleCard';
import reelData from './reel-data.json';

export const FPS = 30;
export const BOOT_FRAMES = 80;
export const TITLE_FRAMES = 100;
export const END_FRAMES = 110;
export const CROSS = 10;

export const scenes: SceneMeta[] = reelData.scenes;

export const sceneFrames = (scene: SceneMeta) =>
  Math.max(CROSS * 2 + 1, Math.floor(scene.durationS * FPS) - 1);

export const totalFrames = () => {
  // Series items after the first start CROSS frames early (crossfade).
  let total = BOOT_FRAMES + TITLE_FRAMES + END_FRAMES;
  for (const scene of scenes) total += sceneFrames(scene);
  total -= CROSS * (scenes.length + 1); // scene overlaps + end card
  return total;
};

export const Reel: React.FC = () => {
  return (
    <AbsoluteFill style={{backgroundColor: '#000'}}>
      <Series>
        <Series.Sequence durationInFrames={BOOT_FRAMES}>
          <BootIntro />
        </Series.Sequence>
        <Series.Sequence durationInFrames={TITLE_FRAMES}>
          <TitleCard />
        </Series.Sequence>
        {scenes.map((scene) => (
          <Series.Sequence
            key={scene.num}
            durationInFrames={sceneFrames(scene)}
            offset={-CROSS}
          >
            <SceneClip scene={scene} frames={sceneFrames(scene)} cross={CROSS} />
          </Series.Sequence>
        ))}
        <Series.Sequence durationInFrames={END_FRAMES} offset={-CROSS}>
          <EndCard />
        </Series.Sequence>
      </Series>
    </AbsoluteFill>
  );
};
