// src/animus/face/faceScreenDraw.js
// Port of the webapp's face screen renderer (k-vrc-webapp
// src/faceScreen.js) for offline frame rendering under Node.
//
// The geometry, palettes, and layer order are kept number for number:
// mood background, weighted expression (drawWeighted), speech
// waveform, glitch slices, CRT scanlines + vignette, LED pixel grid,
// then the blink eyelid overlay. THREE.js texture plumbing and the
// realtime tick loop are replaced by an explicit per-frame state
// object; all randomness comes from an injected rng so a take renders
// byte-identically for a given seed.

import { drawGlyphFace } from './glyphComposer.js';

export const W = 512;
export const H = 512;

// Verbatim from faceScreen.js MOOD_COLORS.
export const MOOD_COLORS = {
  cold:   { primary: '#00cfff', secondary: '#0066aa', bg: '#010810' },
  warm:   { primary: '#ffaa22', secondary: '#ff6600', bg: '#0f0500' },
  glitch: { primary: '#ff00ee', secondary: '#880077', bg: '#080005' },
  static: { primary: '#aaaaaa', secondary: '#555555', bg: '#060606' },
  data:   { primary: '#00ff88', secondary: '#00aa55', bg: '#010a04' },
  boot:   { primary: '#00e5ff', secondary: '#0088aa', bg: '#040c10' },
  angry:  { primary: '#ff2200', secondary: '#881100', bg: '#0f0000' },
  dream:  { primary: '#cc88ff', secondary: '#7744bb', bg: '#070010' },
};

function glow(ctx, color, blur = 20) {
  ctx.shadowColor = color;
  ctx.shadowBlur = blur;
}

function noGlow(ctx) {
  ctx.shadowBlur = 0;
}

// Port of drawCRT: scanlines plus vignette.
function drawCRT(ctx) {
  ctx.globalAlpha = 0.08;
  ctx.fillStyle = '#000000';
  for (let y = 0; y < H; y += 3) {
    ctx.fillRect(0, y, W, 1);
  }
  ctx.globalAlpha = 1;
  const vig = ctx.createRadialGradient(W / 2, H / 2, H * 0.3, W / 2, H / 2, H * 0.75);
  vig.addColorStop(0, 'rgba(0,0,0,0)');
  vig.addColorStop(1, 'rgba(0,0,0,0.65)');
  ctx.fillStyle = vig;
  ctx.fillRect(0, 0, W, H);
}

// Port of applyPixelGrid.
function applyPixelGrid(ctx) {
  ctx.globalAlpha = 0.06;
  ctx.fillStyle = '#000';
  const sz = 8;
  for (let x = 0; x < W; x += sz) {
    for (let y = 0; y < H; y += sz) {
      ctx.fillRect(x, y, 1, sz);
      ctx.fillRect(x, y, sz, 1);
    }
  }
  ctx.globalAlpha = 1;
}

// --- v2 screen structure (reel-polish brief, directive 3) ----------------

// Visible LED cell structure: a much stronger grid than the v1 hint,
// plus the vertical scanline texture reference 4's faces show.
function applyLedGridV2(ctx) {
  ctx.save();
  ctx.fillStyle = '#000';
  const sz = 8;
  ctx.globalAlpha = 0.30;
  for (let x = 0; x < W; x += sz) ctx.fillRect(x, 0, 2, H);
  for (let y = 0; y < H; y += sz) ctx.fillRect(0, y, 2, W);
  // Vertical sub-scanline inside each cell.
  ctx.globalAlpha = 0.10;
  for (let x = 4; x < W; x += sz) ctx.fillRect(x, 0, 1, H);
  ctx.restore();
}

// Soft phosphor bloom: the canvas composited over itself, blurred and
// lightened. Deterministic: pure function of the pixels already drawn.
function applyBloom(ctx) {
  ctx.save();
  ctx.globalCompositeOperation = 'lighter';
  ctx.globalAlpha = 0.4;
  ctx.filter = 'blur(7px)';
  ctx.drawImage(ctx.canvas, 0, 0);
  ctx.filter = 'none';
  ctx.restore();
}

// Slight chromatic fringe: the red channel sampled a step left, the
// blue channel a step right, mixed into the base. Manual pixel walk so
// Node and the browser produce the same bytes.
function applyChromaticFringe(ctx, shift = 2, mix = 0.4) {
  const image = ctx.getImageData(0, 0, W, H);
  const src = image.data;
  const out = new Uint8ClampedArray(src);
  for (let y = 0; y < H; y++) {
    const row = y * W * 4;
    for (let x = 0; x < W; x++) {
      const i = row + x * 4;
      const xr = Math.min(W - 1, Math.max(0, x - shift));
      const xb = Math.min(W - 1, Math.max(0, x + shift));
      out[i] = src[i] * (1 - mix) + src[row + xr * 4] * mix;
      out[i + 2] = src[i + 2] * (1 - mix) + src[row + xb * 4 + 2] * mix;
    }
  }
  image.data.set(out);
  ctx.putImageData(image, 0, 0);
}

// Port of drawWeighted: the six-weight expression face.
function drawWeighted(ctx, weights, c, blink, amplitude, rng) {
  const EY = H * 0.42, EX_OFF = W * 0.175, ES = W * 0.12;

  ctx.fillStyle = c.primary;
  ctx.strokeStyle = c.primary;

  // Eyes: squint compresses eye height
  const eyeH = ES * 1.6 * (1 - weights.eye_squint * 0.75) * (1 - blink);
  [W / 2 - EX_OFF, W / 2 + EX_OFF].forEach((ex) => {
    ctx.beginPath();
    ctx.ellipse(ex, EY, ES * 0.5, Math.max(eyeH * 0.5, 2), 0, 0, Math.PI * 2);
    ctx.fill();
  });

  // Brow: brow_raise moves brows up; brow_furrow angles them inward
  const browY = EY - ES * 0.9 - weights.brow_raise * ES * 0.6;
  ctx.lineWidth = 8;
  ctx.lineCap = 'round';
  if (weights.brow_furrow > 0.15) {
    const furrX = weights.brow_furrow * ES * 0.5;
    ctx.beginPath();
    ctx.moveTo(W / 2 - EX_OFF - ES * 0.5, browY - furrX * 0.5);
    ctx.lineTo(W / 2 - EX_OFF + ES * 0.4, browY + furrX * 0.5);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(W / 2 + EX_OFF + ES * 0.5, browY - furrX * 0.5);
    ctx.lineTo(W / 2 + EX_OFF - ES * 0.4, browY + furrX * 0.5);
    ctx.stroke();
  } else {
    [W / 2 - EX_OFF, W / 2 + EX_OFF].forEach((ex) => {
      ctx.beginPath();
      ctx.moveTo(ex - ES * 0.5, browY);
      ctx.lineTo(ex + ES * 0.5, browY);
      ctx.stroke();
    });
  }

  // Mouth: live amplitude expands mouth_open during speech
  const MY = H * 0.65;
  const liveMouthOpen = Math.max(weights.mouth_open, amplitude * 0.9);
  ctx.lineWidth = 7;
  ctx.lineCap = 'round';
  if (liveMouthOpen > 0.1) {
    ctx.beginPath();
    ctx.ellipse(W / 2, MY, 35 + weights.smile_width * 20, 12 + liveMouthOpen * 20, 0, 0, Math.PI * 2);
    ctx.fill();
  } else if (weights.smile_width > 0.05) {
    ctx.beginPath();
    ctx.arc(W / 2, MY - 10, 25 + weights.smile_width * 30, 0.1 * Math.PI, 0.9 * Math.PI);
    ctx.stroke();
  } else {
    ctx.globalAlpha = 0.6;
    ctx.beginPath();
    ctx.moveTo(W / 2 - 22, MY);
    ctx.lineTo(W / 2 + 22, MY);
    ctx.stroke();
    ctx.globalAlpha = 1;
  }

  // Glitch overlay
  if (weights.glitch_intensity > 0.05) {
    noGlow(ctx);
    const intensity = weights.glitch_intensity;
    for (let i = 0; i < Math.floor(intensity * 8); i++) {
      ctx.globalAlpha = 0.3 + rng() * 0.4;
      ctx.fillStyle = c.primary;
      ctx.fillRect(0, rng() * H, W, 2 + rng() * 3);
    }
    ctx.globalAlpha = 1;
    ctx.fillStyle = c.primary;
    ctx.strokeStyle = c.primary;
  }
}

// Port of drawWaveform.
function drawWaveform(ctx, amp, t, c) {
  if (amp < 0.02) return;
  ctx.save();
  ctx.globalAlpha = 0.75 * amp;
  ctx.strokeStyle = c.primary;
  ctx.lineWidth = 4;
  ctx.lineCap = 'round';
  glow(ctx, c.primary, 14);
  const bars = 14, bw = 12, gap = 8;
  const totalW = bars * (bw + gap);
  const startX = W / 2 - totalW / 2;
  const baseY = H * 0.82;
  for (let i = 0; i < bars; i++) {
    const phase = t * 10 + i * 0.7;
    const h = 6 + amp * 38 * (0.4 + 0.6 * Math.abs(Math.sin(phase)));
    const x = startX + i * (bw + gap);
    ctx.beginPath();
    ctx.moveTo(x + bw / 2, baseY - h / 2);
    ctx.lineTo(x + bw / 2, baseY + h / 2);
    ctx.stroke();
  }
  ctx.restore();
  noGlow(ctx);
}

// Port of drawGlitch (the rare full-screen slice glitch).
function drawGlitch(ctx, c, rng) {
  const slices = 4 + Math.floor(rng() * 5);
  for (let i = 0; i < slices; i++) {
    const y = rng() * H;
    const h = 2 + rng() * 12;
    const shift = (rng() - 0.5) * 30;
    const imgData = ctx.getImageData(0, y, W, h);
    ctx.putImageData(imgData, shift, y);
  }
  ctx.globalAlpha = 0.08;
  ctx.fillStyle = c.primary;
  ctx.fillRect(rng() * W * 0.3, rng() * H, W * 0.7, 2 + rng() * 4);
  ctx.globalAlpha = 1;
}

// Port of drawBlink: eyelid slit in the background color.
function drawBlink(ctx, p, c) {
  const EY = H * 0.42, EX_OFF = W * 0.175, ES = W * 0.12;
  ctx.fillStyle = c.bg;
  glow(ctx, c.primary, 6);
  [W / 2 - EX_OFF, W / 2 + EX_OFF].forEach((ex) => {
    const closeH = ES * 1.8 * p;
    ctx.fillRect(ex - ES * 0.6, EY - closeH / 2, ES * 1.2, closeH);
  });
  noGlow(ctx);
}

// Port of the main draw() minus the boot sequence: one finished take
// frame from one timeline state.
//
// state: { t, weights, mood, amplitude, blinkProgress, glitchActive }
//        plus optional glyph (a validated glyph spec: the face is a
//        composed glyph instead of a weighted library expression)
// rng: seeded 0..1 generator (replaces Math.random)
// opts: { fx } - the v2 screen pass (strong LED grid, bloom,
//        chromatic fringe) defaults ON; pass fx: false for the flat
//        v1 look.
export function drawFaceFrame(ctx, state, rng, opts = {}) {
  const fx = opts.fx !== false;
  const c = MOOD_COLORS[state.mood] ?? MOOD_COLORS.cold;
  const amplitude = state.amplitude;
  const glyph = state.glyph ?? null;
  const textMode = Boolean(glyph && glyph.text);

  // Background
  ctx.fillStyle = c.bg;
  ctx.fillRect(0, 0, W, H);

  // Expression, with the subtle scale bob speech amplitude adds
  ctx.save();
  if (amplitude > 0.02) {
    const bob = 1 + amplitude * 0.016;
    ctx.translate(W / 2, H / 2);
    ctx.scale(bob, bob);
    ctx.translate(-W / 2, -H / 2);
  }
  ctx.textBaseline = 'middle';
  ctx.textAlign = 'center';
  ctx.fillStyle = c.primary;
  ctx.strokeStyle = c.primary;
  glow(ctx, c.primary, 24);
  if (glyph) {
    drawGlyphFace(ctx, glyph, c, textMode ? 0 : state.blinkProgress, amplitude);
  } else {
    drawWeighted(ctx, state.weights, c, state.blinkProgress, amplitude, rng);
  }
  noGlow(ctx);
  ctx.restore();

  // Speaking waveform
  if (amplitude > 0.02) drawWaveform(ctx, amplitude, state.t, c);

  // Rare glitch
  if (state.glitchActive) drawGlitch(ctx, c, rng);

  // v2 screen structure: fringe and bloom act on the drawn content,
  // then the CRT layers and LED grid sit on top of everything.
  if (fx) {
    applyChromaticFringe(ctx);
    applyBloom(ctx);
  }
  drawCRT(ctx);
  applyPixelGrid(ctx);
  if (fx) applyLedGridV2(ctx);

  // Blink overlay (a text face has no eyes to blink)
  if (state.blinkProgress > 0 && !textMode) drawBlink(ctx, state.blinkProgress, c);
}
