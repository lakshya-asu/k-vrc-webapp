import React, {useEffect, useRef} from 'react';
import {Easing, interpolate, useCurrentFrame, useVideoConfig} from 'remotion';
// @ts-ignore
import {mulberry32} from '../../../src/animus/face/faceTimeline.js';
import {LED_WARM, drawLedText, drawScanlines, drawVignette} from './led';

// "PROJECT ANIMUS" in the LED screen aesthetic: dot-matrix reveal,
// scanlines, glow, vignette. Drawn per frame on a full-HD canvas.
export const TitleCard: React.FC = () => {
  const frame = useCurrentFrame();
  const {width, height, durationInFrames} = useVideoConfig();
  const ref = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.fillStyle = LED_WARM.bg;
    ctx.fillRect(0, 0, width, height);

    const rng = mulberry32((9001 + frame) >>> 0);
    const reveal = interpolate(frame, [6, 34], [0, 1], {
      extrapolateLeft: 'clamp',
      extrapolateRight: 'clamp',
    });
    drawLedText(ctx, 'PROJECT ANIMUS', {
      x: width * 0.08,
      y: height * 0.44,
      width: width * 0.84,
      palette: LED_WARM,
      cell: 9,
      font: '900 160px "Arial Black", Arial, sans-serif',
      reveal,
      jitter: rng,
    });

    const subAlpha = interpolate(frame, [38, 56], [0, 1], {
      extrapolateLeft: 'clamp',
      extrapolateRight: 'clamp',
    });
    ctx.save();
    ctx.globalAlpha = subAlpha;
    ctx.fillStyle = LED_WARM.primary;
    ctx.shadowColor = LED_WARM.primary;
    ctx.shadowBlur = 18;
    ctx.font = '400 44px "Consolas", "Courier New", monospace';
    ctx.textAlign = 'center';
    ctx.fillText('an autonomous AI actor', width / 2, height * 0.62);
    ctx.restore();

    drawScanlines(ctx, width, height);
    drawVignette(ctx, width, height);

    // Fade the whole card gently out over its last 18 frames (eased,
    // overlapping the first scene's eased fade-in).
    const fadeOut = interpolate(
      frame,
      [durationInFrames - 18, durationInFrames - 1],
      [0, 1],
      {
        extrapolateLeft: 'clamp',
        extrapolateRight: 'clamp',
        easing: Easing.inOut(Easing.quad),
      },
    );
    if (fadeOut > 0) {
      ctx.fillStyle = `rgba(0, 0, 0, ${fadeOut})`;
      ctx.fillRect(0, 0, width, height);
    }
  }, [frame, width, height, durationInFrames]);

  return (
    <canvas
      ref={ref}
      width={width}
      height={height}
      style={{width: '100%', height: '100%'}}
    />
  );
};
