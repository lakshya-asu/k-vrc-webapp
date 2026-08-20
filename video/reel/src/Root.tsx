import React from 'react';
import {Composition} from 'remotion';
import {FPS, Reel, totalFrames} from './Reel';

export const Root: React.FC = () => {
  return (
    <Composition
      id="ProjectAnimusReel"
      component={Reel}
      durationInFrames={totalFrames()}
      fps={FPS}
      width={1920}
      height={1080}
    />
  );
};
