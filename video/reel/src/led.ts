// LED / CRT aesthetic helpers for the title and caption cards.
//
// Visual language matches the K-VRC visor face renderer
// (src/animus/face/faceScreenDraw.js): dark ground, glowing primary
// color, horizontal scanlines, a coarse LED pixel grid, and a soft
// vignette. The face code itself stays untouched; these helpers only
// reproduce its look for typography.

export type LedPalette = {
  bg: string;
  primary: string;
  dim: string;
};

export const LED_WARM: LedPalette = {
  bg: '#180d02',
  primary: '#ffb347',
  dim: 'rgba(255, 179, 71, 0.35)',
};

export const LED_COLD: LedPalette = {
  bg: '#020a12',
  primary: '#69d2ff',
  dim: 'rgba(105, 210, 255, 0.35)',
};

export function drawScanlines(
  ctx: CanvasRenderingContext2D,
  w: number,
  h: number,
  alpha = 0.16,
  step = 4,
) {
  ctx.save();
  ctx.fillStyle = `rgba(0, 0, 0, ${alpha})`;
  for (let y = 0; y < h; y += step) {
    ctx.fillRect(0, y, w, Math.max(1, Math.floor(step / 2)));
  }
  ctx.restore();
}

export function drawVignette(
  ctx: CanvasRenderingContext2D,
  w: number,
  h: number,
) {
  const grad = ctx.createRadialGradient(
    w / 2,
    h / 2,
    Math.min(w, h) * 0.35,
    w / 2,
    h / 2,
    Math.max(w, h) * 0.72,
  );
  grad.addColorStop(0, 'rgba(0,0,0,0)');
  grad.addColorStop(1, 'rgba(0,0,0,0.55)');
  ctx.save();
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, w, h);
  ctx.restore();
}

// Render text as a coarse LED dot matrix: rasterize to an offscreen
// canvas, then sample it into round glowing dots on the target.
export function drawLedText(
  ctx: CanvasRenderingContext2D,
  text: string,
  options: {
    x: number;
    y: number;
    width: number;
    palette: LedPalette;
    cell?: number;
    font?: string;
    reveal?: number; // 0..1, columns revealed left to right
    jitter?: () => number; // deterministic rng for flicker
  },
) {
  const cell = options.cell ?? 10;
  const font = options.font ?? '900 120px "Arial Black", Arial, sans-serif';
  const reveal = options.reveal ?? 1;
  const off = document.createElement('canvas');
  const measure = off.getContext('2d')!;
  measure.font = font;
  const metrics = measure.measureText(text);
  const textW = Math.ceil(metrics.width) + cell * 4;
  const textH = Math.ceil(
    (metrics.actualBoundingBoxAscent || 90) +
      (metrics.actualBoundingBoxDescent || 30),
  ) + cell * 4;
  off.width = textW;
  off.height = textH;
  const octx = off.getContext('2d')!;
  octx.font = font;
  octx.fillStyle = '#ffffff';
  octx.textBaseline = 'middle';
  octx.textAlign = 'center';
  octx.fillText(text, textW / 2, textH / 2);
  const data = octx.getImageData(0, 0, textW, textH).data;

  const scale = options.width / textW;
  const drawnH = textH * scale;
  const originX = options.x;
  const originY = options.y - drawnH / 2;

  ctx.save();
  ctx.fillStyle = options.palette.primary;
  ctx.shadowColor = options.palette.primary;
  ctx.shadowBlur = cell * scale * 1.6;
  const cols = Math.ceil(textW / cell);
  const revealCols = Math.ceil(cols * reveal);
  for (let cy = 0; cy < textH; cy += cell) {
    for (let cx = 0; cx < textW; cx += cell) {
      if (cx / cell >= revealCols) continue;
      // Sample the cell center's alpha.
      const px = Math.min(textW - 1, cx + Math.floor(cell / 2));
      const py = Math.min(textH - 1, cy + Math.floor(cell / 2));
      const alpha = data[(py * textW + px) * 4 + 3] / 255;
      if (alpha < 0.4) continue;
      const flicker = options.jitter ? 0.88 + options.jitter() * 0.12 : 1;
      ctx.globalAlpha = alpha * flicker;
      const dotR = (cell * scale) * 0.38;
      ctx.beginPath();
      ctx.arc(
        originX + (cx + cell / 2) * scale,
        originY + (cy + cell / 2) * scale,
        dotR,
        0,
        Math.PI * 2,
      );
      ctx.fill();
    }
  }
  ctx.restore();
  return {height: drawnH};
}
