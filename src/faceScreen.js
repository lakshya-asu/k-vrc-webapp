import * as THREE from 'three';

// ── Canvas setup ─────────────────────────────────────────────
const W = 512, H = 512;
const canvas = document.createElement('canvas');
canvas.width = W; canvas.height = H;
const ctx = canvas.getContext('2d', { willReadFrequently: true });

export const faceTexture = new THREE.CanvasTexture(canvas);
faceTexture.minFilter = THREE.NearestFilter;
faceTexture.magFilter = THREE.NearestFilter;

// ── State ────────────────────────────────────────────────────
let currentEmotion = 'neutral';
let t = 0;                     // global time (seconds)
let amplitude = 0;             // speech amplitude 0-1
let blinkT = randomBetween(3, 7);
let blinking = false;
let blinkProgress = 0;         // 0=open, 1=closed, animates via tickFaceScreen
let glitchTimer = 0;
let glitchActive = false;
let booting = true;
let bootProgress = 0;          // 0→1 over ~2s

// Morph state — smoothly blend eye shapes
let morphFrom = null;
let morphTo = null;
let morphP = 1;               // 0=from, 1=to (complete)
const MORPH_SPEED = 4;        // units/sec

// faceMaterial reference set by attachFaceScreen
let faceMaterial = null;
let fadeAnimId = null;
let _expressionWeights = null; // null = use emotion draw path
let _expressionMood = null;    // null = use emotion color path

function randomBetween(a, b) { return a + Math.random() * (b - a); }

// ── Color palette ────────────────────────────────────────────
const COLORS = {
  neutral:  { primary: '#00e5ff', secondary: '#0088aa', bg: '#040c10' },
  happy:    { primary: '#00ff88', secondary: '#008844', bg: '#020e06' },
  sad:      { primary: '#4488ff', secondary: '#224488', bg: '#02040e' },
  excited:  { primary: '#ffe600', secondary: '#ff8800', bg: '#100a00' },
  thinking: { primary: '#bb44ff', secondary: '#661188', bg: '#080010' },
  angry:    { primary: '#ff2200', secondary: '#881100', bg: '#100000' },
};

const MOOD_COLORS = {
  cold:   { primary: '#00cfff', secondary: '#0066aa', bg: '#010810' },
  warm:   { primary: '#ffaa22', secondary: '#ff6600', bg: '#0f0500' },
  glitch: { primary: '#ff00ee', secondary: '#880077', bg: '#080005' },
  static: { primary: '#aaaaaa', secondary: '#555555', bg: '#060606' },
  data:   { primary: '#00ff88', secondary: '#00aa55', bg: '#010a04' },
  boot:   { primary: '#00e5ff', secondary: '#0088aa', bg: '#040c10' },
  angry:  { primary: '#ff2200', secondary: '#881100', bg: '#0f0000' },
  dream:  { primary: '#cc88ff', secondary: '#7744bb', bg: '#070010' },
};

function getColors() {
  if (_expressionMood) return MOOD_COLORS[_expressionMood] ?? MOOD_COLORS.cold;
  return COLORS[currentEmotion] ?? COLORS.neutral;
}

// ── Eye shape definitions ─────────────────────────────────────
// Each eye is defined as a draw function: (ctx, cx, cy, w, h, color, blink)
// blink: 0=fully open, 1=fully closed
const EYES = {
  neutral(ctx, cx, cy, w, h, col, blink) {
    const eh = h * (1 - blink);
    ctx.beginPath();
    ctx.ellipse(cx, cy, w * 0.5, eh * 0.5, 0, 0, Math.PI * 2);
    ctx.fill();
  },
  happy(ctx, cx, cy, w, h, col, blink) {
    // ^^ arcs (kawaii style)
    const eh = h * (1 - blink);
    ctx.lineWidth = 10;
    ctx.beginPath();
    ctx.arc(cx, cy + eh * 0.3, w * 0.42, Math.PI, 0);
    ctx.stroke();
  },
  sad(ctx, cx, cy, w, h, col, blink) {
    // droopy half-circle
    const eh = h * (1 - blink);
    ctx.lineWidth = 10;
    ctx.beginPath();
    ctx.arc(cx, cy - eh * 0.1, w * 0.4, 0, Math.PI);
    ctx.stroke();
  },
  excited(ctx, cx, cy, w, h, col, blink) {
    // stars
    const s = w * 0.45 * (1 - blink * 0.9);
    drawStar(ctx, cx, cy, s * 0.4, s, 5);
    ctx.fill();
  },
  thinking(ctx, cx, cy, w, h, col, blink) {
    // one squinting horizontal slit
    const eh = h * 0.15 * (1 - blink);
    ctx.beginPath();
    ctx.ellipse(cx, cy, w * 0.5, Math.max(eh, 2), 0, 0, Math.PI * 2);
    ctx.fill();
  },
  angry(ctx, cx, cy, w, h, col, blink) {
    // sharp angled slit
    const eh = h * 0.2 * (1 - blink);
    ctx.save();
    const angle = cx < W / 2 ? 0.35 : -0.35; // inner corners up
    ctx.translate(cx, cy);
    ctx.rotate(angle);
    ctx.beginPath();
    ctx.ellipse(0, 0, w * 0.5, Math.max(eh, 3), 0, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();
  },
};

function drawStar(ctx, cx, cy, r1, r2, pts) {
  ctx.beginPath();
  for (let i = 0; i < pts * 2; i++) {
    const a = (i * Math.PI) / pts - Math.PI / 2;
    const r = i % 2 === 0 ? r2 : r1;
    i === 0 ? ctx.moveTo(cx + Math.cos(a) * r, cy + Math.sin(a) * r)
            : ctx.lineTo(cx + Math.cos(a) * r, cy + Math.sin(a) * r);
  }
  ctx.closePath();
}

// ── Scanline + CRT overlay ────────────────────────────────────
function drawCRT() {
  // Scanlines
  ctx.globalAlpha = 0.08;
  ctx.fillStyle = '#000000';
  for (let y = 0; y < H; y += 3) {
    ctx.fillRect(0, y, W, 1);
  }
  // Vignette
  ctx.globalAlpha = 1;
  const vig = ctx.createRadialGradient(W/2, H/2, H*0.3, W/2, H/2, H*0.75);
  vig.addColorStop(0, 'rgba(0,0,0,0)');
  vig.addColorStop(1, 'rgba(0,0,0,0.65)');
  ctx.fillStyle = vig;
  ctx.fillRect(0, 0, W, H);
}

// ── LED pixel grid ────────────────────────────────────────────
function applyPixelGrid() {
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

// ── Glow helper ───────────────────────────────────────────────
function glow(color, blur = 20) {
  ctx.shadowColor = color;
  ctx.shadowBlur = blur;
}
function noGlow() { ctx.shadowBlur = 0; }

// ── Expression renderers ─────────────────────────────────────
const EXPRESSIONS = {
  neutral(c, blink) {
    const EY = H * 0.42, EX_OFF = W * 0.175, ES = W * 0.12;
    ctx.fillStyle = c.primary;
    ctx.strokeStyle = c.primary;
    glow(c.primary, 28);
    [W/2 - EX_OFF, W/2 + EX_OFF].forEach(ex => {
      EYES.neutral(ctx, ex, EY, ES*2, ES*1.6, c.primary, blink);
    });
    noGlow();
    // subtle mouth line
    ctx.globalAlpha = 0.6;
    ctx.lineWidth = 5;
    ctx.lineCap = 'round';
    ctx.beginPath();
    ctx.moveTo(W/2 - 36, H * 0.64);
    ctx.lineTo(W/2 + 36, H * 0.64);
    ctx.stroke();
    ctx.globalAlpha = 1;
  },

  happy(c, blink) {
    const EY = H * 0.44, EX_OFF = W * 0.175, ES = W * 0.13;
    ctx.fillStyle = c.primary;
    ctx.strokeStyle = c.primary;
    glow(c.primary, 30);
    [W/2 - EX_OFF, W/2 + EX_OFF].forEach(ex => {
      EYES.happy(ctx, ex, EY, ES*2, ES*1.6, c.primary, blink);
    });
    // Smile
    ctx.lineWidth = 9;
    ctx.lineCap = 'round';
    ctx.beginPath();
    ctx.arc(W/2, H * 0.60, 48, 0.08*Math.PI, 0.92*Math.PI);
    ctx.stroke();
    // Cheek blush dots
    noGlow();
    ctx.globalAlpha = 0.35;
    ctx.fillStyle = c.primary;
    ctx.beginPath(); ctx.arc(W/2 - 105, H*0.56, 14, 0, Math.PI*2); ctx.fill();
    ctx.beginPath(); ctx.arc(W/2 + 105, H*0.56, 14, 0, Math.PI*2); ctx.fill();
    ctx.globalAlpha = 1;
  },

  sad(c, blink) {
    const EY = H * 0.42, EX_OFF = W * 0.175, ES = W * 0.12;
    ctx.fillStyle = c.primary;
    ctx.strokeStyle = c.primary;
    glow(c.primary, 22);
    [W/2 - EX_OFF, W/2 + EX_OFF].forEach(ex => {
      EYES.sad(ctx, ex, EY, ES*2, ES*1.6, c.primary, blink);
    });
    // Tear drop left eye
    if (!blinking) {
      ctx.globalAlpha = 0.7;
      const tx = W/2 - EX_OFF + 6;
      ctx.beginPath();
      ctx.moveTo(tx, EY + 18);
      ctx.bezierCurveTo(tx - 7, EY + 32, tx - 7, EY + 44, tx, EY + 48);
      ctx.bezierCurveTo(tx + 7, EY + 44, tx + 7, EY + 32, tx, EY + 18);
      ctx.fill();
      ctx.globalAlpha = 1;
    }
    // Frown
    ctx.lineWidth = 8; ctx.lineCap = 'round';
    ctx.beginPath();
    ctx.arc(W/2, H*0.70, 40, 1.1*Math.PI, 1.9*Math.PI);
    ctx.stroke();
    noGlow();
  },

  excited(c, blink) {
    const EY = H * 0.41, EX_OFF = W * 0.175, ES = W * 0.13;
    const pulse = 0.85 + 0.15 * Math.sin(t * 7);
    ctx.fillStyle = c.primary;
    ctx.strokeStyle = c.primary;
    glow(c.primary, 35);
    [W/2 - EX_OFF, W/2 + EX_OFF].forEach(ex => {
      ctx.save();
      ctx.translate(ex, EY);
      ctx.scale(pulse, pulse);
      ctx.translate(-ex, -EY);
      EYES.excited(ctx, ex, EY, ES*2, ES*2, c.primary, blink);
      ctx.restore();
    });
    // Big open mouth
    ctx.lineWidth = 10; ctx.lineCap = 'round';
    ctx.beginPath();
    ctx.arc(W/2, H*0.58, 55, 0.05*Math.PI, 0.95*Math.PI);
    ctx.stroke();
    // "!!" text
    noGlow(); glow(c.secondary, 12);
    ctx.font = `bold ${W*0.12}px monospace`;
    ctx.textAlign = 'center';
    ctx.fillStyle = c.primary;
    ctx.globalAlpha = 0.9;
    ctx.fillText('!!', W/2, H*0.82);
    ctx.globalAlpha = 1;
  },

  thinking(c, blink) {
    const EY = H * 0.42, EX_OFF = W * 0.175, ES = W * 0.12;
    ctx.fillStyle = c.primary;
    ctx.strokeStyle = c.primary;
    glow(c.primary, 20);
    // Left eye: squint; Right eye: looking up-right (shifted)
    EYES.thinking(ctx, W/2 - EX_OFF, EY, ES*2, ES*1.6, c.primary, blink);
    EYES.neutral(ctx, W/2 + EX_OFF - 8, EY - 8, ES*1.6, ES*1.2, c.primary, blink);
    // Animated "..." dots
    noGlow();
    const dotCount = Math.floor(t * 1.8) % 4;
    ctx.font = `bold ${W*0.15}px monospace`;
    ctx.textAlign = 'center';
    ctx.globalAlpha = 0.9;
    ctx.fillText('.'.repeat(dotCount + 1), W/2, H*0.72);
    ctx.globalAlpha = 1;
  },

  angry(c, blink) {
    const EY = H * 0.42, EX_OFF = W * 0.175, ES = W * 0.12;
    ctx.fillStyle = c.primary;
    ctx.strokeStyle = c.primary;
    glow(c.primary, 28);
    // Angled brow lines
    ctx.lineWidth = 10; ctx.lineCap = 'round';
    ctx.beginPath(); ctx.moveTo(W/2-EX_OFF-ES, EY-ES*0.8); ctx.lineTo(W/2-EX_OFF+ES*0.3, EY-ES*0.2); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(W/2+EX_OFF+ES, EY-ES*0.8); ctx.lineTo(W/2+EX_OFF-ES*0.3, EY-ES*0.2); ctx.stroke();
    [W/2 - EX_OFF, W/2 + EX_OFF].forEach(ex => {
      EYES.angry(ctx, ex, EY, ES*2, ES*1.2, c.primary, blink);
    });
    // Jagged mouth
    noGlow(); glow(c.primary, 16);
    ctx.lineWidth = 8;
    ctx.beginPath();
    const mx = W/2, my = H*0.66, mw = 90;
    ctx.moveTo(mx - mw/2, my);
    for (let i = 0; i <= 4; i++) {
      ctx.lineTo(mx - mw/2 + i*(mw/4), my + (i%2===0 ? 0 : 20));
    }
    ctx.stroke();
  },
};

// ── Speaking waveform overlay ─────────────────────────────────
function drawWaveform(amp) {
  if (amp < 0.02) return;
  const c = getColors();
  ctx.save();
  ctx.globalAlpha = 0.75 * amp;
  ctx.strokeStyle = c.primary;
  ctx.lineWidth = 4;
  ctx.lineCap = 'round';
  glow(c.primary, 14);
  const bars = 14, bw = 12, gap = 8;
  const totalW = bars * (bw + gap);
  const startX = W/2 - totalW/2;
  const baseY = H * 0.82;
  for (let i = 0; i < bars; i++) {
    const phase = t * 10 + i * 0.7;
    const h = 6 + amp * 38 * (0.4 + 0.6 * Math.abs(Math.sin(phase)));
    const x = startX + i * (bw + gap);
    ctx.beginPath();
    ctx.moveTo(x + bw/2, baseY - h/2);
    ctx.lineTo(x + bw/2, baseY + h/2);
    ctx.stroke();
  }
  ctx.restore();
  noGlow();
}

// ── Glitch effect ─────────────────────────────────────────────
function drawGlitch() {
  const slices = 4 + Math.floor(Math.random() * 5);
  for (let i = 0; i < slices; i++) {
    const y = Math.random() * H;
    const h = 2 + Math.random() * 12;
    const shift = (Math.random() - 0.5) * 30;
    const imgData = ctx.getImageData(0, y, W, h);
    ctx.putImageData(imgData, shift, y);
  }
  // Random color channel ghost
  const c = getColors();
  ctx.globalAlpha = 0.08;
  ctx.fillStyle = c.primary;
  ctx.fillRect(Math.random() * W * 0.3, Math.random() * H, W * 0.7, 2 + Math.random() * 4);
  ctx.globalAlpha = 1;
}

// ── Boot sequence ─────────────────────────────────────────────
function drawBoot(p) {
  const c = COLORS.neutral;
  ctx.fillStyle = '#000';
  ctx.fillRect(0, 0, W, H);

  if (p < 0.2) {
    // Flicker / static
    ctx.globalAlpha = p / 0.2;
    for (let i = 0; i < 40; i++) {
      ctx.fillStyle = Math.random() > 0.5 ? c.primary : '#000';
      ctx.fillRect(Math.random()*W, Math.random()*H, 8, 3);
    }
    ctx.globalAlpha = 1;
    return;
  }

  if (p < 0.5) {
    // Loading bar
    const prog = (p - 0.2) / 0.3;
    ctx.strokeStyle = c.primary;
    ctx.lineWidth = 3;
    ctx.strokeRect(W*0.2, H*0.55, W*0.6, 12);
    glow(c.primary, 18);
    ctx.fillStyle = c.primary;
    ctx.fillRect(W*0.2 + 2, H*0.55 + 2, (W*0.6 - 4) * prog, 8);
    noGlow();
    ctx.font = `${W*0.06}px monospace`;
    ctx.textAlign = 'center';
    ctx.fillStyle = c.primary;
    ctx.globalAlpha = 0.8;
    ctx.fillText('INITIALIZING...', W/2, H*0.47);
    ctx.globalAlpha = 1;
    return;
  }

  // Eyes opening from slits
  const openP = Math.min((p - 0.5) / 0.35, 1);
  ctx.fillStyle = '#000204';
  ctx.fillRect(0, 0, W, H);
  drawCRT();

  const EY = H*0.42, EX_OFF = W*0.175, ES = W*0.12;
  ctx.fillStyle = c.primary;
  glow(c.primary, 22 * openP);
  [W/2 - EX_OFF, W/2 + EX_OFF].forEach(ex => {
    const eh = ES * 1.6 * openP;
    ctx.beginPath();
    ctx.ellipse(ex, EY, ES*0.5, eh*0.5, 0, 0, Math.PI*2);
    ctx.fill();
  });
  noGlow();

  if (p > 0.85) {
    const waveP = (p - 0.85) / 0.15;
    ctx.globalAlpha = waveP * 0.8;
    ctx.strokeStyle = c.primary;
    ctx.lineWidth = 5;
    ctx.lineCap = 'round';
    glow(c.primary, 16);
    ctx.beginPath();
    ctx.moveTo(W/2 - 20, H * 0.75);
    ctx.lineTo(W/2 + 20, H * 0.75);
    ctx.stroke();
    noGlow();
    ctx.globalAlpha = 1;
  }
}

// ── Blink draw ────────────────────────────────────────────────
function drawBlink(p) {
  // p: 0=open, 1=closed. Draws eyelid close over top of current expression.
  const EY = H*0.42, EX_OFF = W*0.175, ES = W*0.12;
  const c = getColors();
  ctx.fillStyle = getColors().bg;
  glow(c.primary, 6);
  [W/2 - EX_OFF, W/2 + EX_OFF].forEach(ex => {
    // Cover eye with bg-color slit
    const closeH = ES * 1.8 * p;
    ctx.fillRect(ex - ES*0.6, EY - closeH/2, ES*1.2, closeH);
  });
  noGlow();
}

// ── Weighted expression draw ──────────────────────────────────
function drawWeighted(weights, c, blink) {
  const EY = H * 0.42, EX_OFF = W * 0.175, ES = W * 0.12;

  ctx.fillStyle = c.primary;
  ctx.strokeStyle = c.primary;

  // Eyes: squint compresses eye height
  const eyeH = ES * 1.6 * (1 - weights.eye_squint * 0.75) * (1 - blink);
  [W/2 - EX_OFF, W/2 + EX_OFF].forEach(ex => {
    ctx.beginPath();
    ctx.ellipse(ex, EY, ES * 0.5, Math.max(eyeH * 0.5, 2), 0, 0, Math.PI * 2);
    ctx.fill();
  });

  // Brow: brow_raise moves brows up; brow_furrow angles them inward
  const browY = EY - ES * 0.9 - weights.brow_raise * ES * 0.6;
  ctx.lineWidth = 8; ctx.lineCap = 'round';
  if (weights.brow_furrow > 0.15) {
    const furrX = weights.brow_furrow * ES * 0.5;
    ctx.beginPath();
    ctx.moveTo(W/2 - EX_OFF - ES * 0.5, browY - furrX * 0.5);
    ctx.lineTo(W/2 - EX_OFF + ES * 0.4, browY + furrX * 0.5);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(W/2 + EX_OFF + ES * 0.5, browY - furrX * 0.5);
    ctx.lineTo(W/2 + EX_OFF - ES * 0.4, browY + furrX * 0.5);
    ctx.stroke();
  } else {
    [W/2 - EX_OFF, W/2 + EX_OFF].forEach(ex => {
      ctx.beginPath();
      ctx.moveTo(ex - ES * 0.5, browY);
      ctx.lineTo(ex + ES * 0.5, browY);
      ctx.stroke();
    });
  }

  // Mouth
  const MY = H * 0.65;
  ctx.lineWidth = 7; ctx.lineCap = 'round';
  if (weights.mouth_open > 0.1) {
    ctx.beginPath();
    ctx.ellipse(W/2, MY, 35 + weights.smile_width * 20, 12 + weights.mouth_open * 20, 0, 0, Math.PI * 2);
    ctx.fill();
  } else if (weights.smile_width > 0.05) {
    ctx.beginPath();
    ctx.arc(W/2, MY - 10, 25 + weights.smile_width * 30, 0.1 * Math.PI, 0.9 * Math.PI);
    ctx.stroke();
  } else {
    ctx.globalAlpha = 0.6;
    ctx.beginPath();
    ctx.moveTo(W/2 - 22, MY);
    ctx.lineTo(W/2 + 22, MY);
    ctx.stroke();
    ctx.globalAlpha = 1;
  }

  // Glitch overlay
  if (weights.glitch_intensity > 0.05) {
    noGlow();
    const intensity = weights.glitch_intensity;
    for (let i = 0; i < Math.floor(intensity * 8); i++) {
      ctx.globalAlpha = 0.3 + Math.random() * 0.4;
      ctx.fillStyle = c.primary;
      ctx.fillRect(0, Math.random() * H, W, 2 + Math.random() * 3);
    }
    ctx.globalAlpha = 1;
    ctx.fillStyle = c.primary;
    ctx.strokeStyle = c.primary;
  }
}

// ── Main draw ─────────────────────────────────────────────────
function draw() {
  const c = getColors();

  if (booting) {
    drawBoot(bootProgress);
    faceTexture.needsUpdate = true;
    return;
  }

  // Background
  ctx.fillStyle = c.bg;
  ctx.fillRect(0, 0, W, H);

  // Expression
  ctx.textBaseline = 'middle';
  ctx.textAlign = 'center';
  ctx.fillStyle = c.primary;
  ctx.strokeStyle = c.primary;
  glow(c.primary, 24);

  if (_expressionWeights) {
    drawWeighted(_expressionWeights, c, blinkProgress);
  } else {
    const expr = EXPRESSIONS[currentEmotion] ?? EXPRESSIONS.neutral;
    expr(c, blinkProgress);
  }
  noGlow();

  // Speaking waveform
  if (amplitude > 0.02) drawWaveform(amplitude);

  // Glitch overlay (random, rare)
  if (glitchActive) drawGlitch();

  // CRT + pixel grid on top
  drawCRT();
  applyPixelGrid();

  // Blink overlay
  if (blinkProgress > 0) drawBlink(blinkProgress);

  faceTexture.needsUpdate = true;
}

// ── Tick (called every frame from robot.js) ───────────────────
export function tickFaceScreen(deltaMs) {
  const delta = deltaMs / 1000;
  t += delta;

  // Boot sequence
  if (booting) {
    bootProgress = Math.min(bootProgress + delta * 0.55, 1);
    if (bootProgress >= 1) {
      booting = false;
      setEmotion('neutral');
    }
    draw();
    return;
  }

  // Morph (not used for geometry right now but reserved for future lerping)
  if (morphP < 1) morphP = Math.min(morphP + delta * MORPH_SPEED, 1);

  // Blink
  blinkT -= delta;
  if (blinkT <= 0 && !blinking) {
    blinking = true;
    blinkT = randomBetween(3.5, 7);
  }
  if (blinking) {
    blinkProgress += delta * 14;  // 0→1 fast
    if (blinkProgress >= 1) {
      blinkProgress = 1 - (blinkProgress - 1); // reverse
      if (blinkProgress <= 0) { blinkProgress = 0; blinking = false; }
    }
  }

  // Rare glitch
  glitchTimer -= delta;
  if (glitchTimer <= 0) {
    glitchTimer = randomBetween(8, 20);
    glitchActive = true;
    setTimeout(() => { glitchActive = false; }, 80 + Math.random() * 120);
  }

  draw();
}

// ── Public API ────────────────────────────────────────────────
export function setEmotion(emotion) {
  _expressionWeights = null;                           // always clear first
  _expressionMood = null;
  if (emotion === currentEmotion && !booting) return;  // guard retained, moved to line 3
  currentEmotion = emotion;
  morphP = 0;
}

export function setExpression(name) {
  if (!setExpression._lib) {
    _expressionWeights = null;
    _expressionMood = null;
    return 0.5;
  }
  const entry = setExpression._lib.find(e => e.name === name)
    ?? setExpression._lib.find(e => e.name === 'neutral_idle');
  if (!entry) {
    _expressionWeights = null;
    _expressionMood = null;
    return 0.5;
  }
  _expressionWeights = { ...entry.weights };
  _expressionMood = entry.mood;
  return entry.motion_energy;
}

export function initExpressionLibrary(lib) {
  setExpression._lib = lib;
}

export function setSpeakingAmplitude(amp) {
  amplitude = Math.max(0, Math.min(1, amp));
}

// ── Attach to mesh ────────────────────────────────────────────
export function attachFaceScreen(robotScene, threeScene, faceBone) {
  const mat = new THREE.MeshStandardMaterial({
    map: faceTexture,
    emissiveMap: faceTexture,
    emissive: new THREE.Color(1, 1, 1),
    emissiveIntensity: 1.4,
    color: new THREE.Color(0, 0, 0),
    transparent: false,
    toneMapped: false,
  });
  faceMaterial = mat;

  // Try to find visor/screen mesh by name or material
  let visorMesh = null;
  robotScene.traverse(obj => {
    if (visorMesh || !obj.isMesh) return;
    const n = obj.name.toLowerCase();
    const matName = (Array.isArray(obj.material) ? obj.material[0] : obj.material)?.name ?? '';
    if (['visor','screen','face','display','screenface'].some(k => n.includes(k) || matName.toLowerCase().includes(k))) {
      visorMesh = obj;
    }
  });

  if (visorMesh) {
    console.log('Face screen: found mesh', visorMesh.name);
    if (Array.isArray(visorMesh.material)) {
      const idx = visorMesh.material.findIndex(m => m?.name?.toLowerCase().includes('screen') || m?.name?.toLowerCase().includes('face'));
      if (idx >= 0) visorMesh.material[idx] = mat;
      else visorMesh.material = mat;
    } else {
      visorMesh.material = mat;
    }
    draw();
    return visorMesh;
  }

  // Fallback floating plane
  console.warn('Face screen: no visor mesh found, using fallback plane');
  const plane = new THREE.Mesh(new THREE.PlaneGeometry(0.3, 0.24), mat);
  plane.__isFallback = true;
  plane.__faceBone = faceBone;
  threeScene.add(plane);
  draw();
  return plane;
}

const _faceOffset = new THREE.Vector3();
export function updateFaceScreen(mesh) {
  if (!mesh?.__isFallback) return;
  const bone = mesh.__faceBone;
  if (!bone) return;
  bone.getWorldPosition(mesh.position);
  bone.getWorldQuaternion(mesh.quaternion);
  _faceOffset.set(0, 0, 0.06).applyQuaternion(mesh.quaternion);
  mesh.position.add(_faceOffset);
  mesh.updateMatrixWorld();
}

// ── Crossfade opacity ─────────────────────────────────────────
export function tweenFaceOpacity(from, to, ms, onDone) {
  if (fadeAnimId) cancelAnimationFrame(fadeAnimId);
  const start = performance.now();
  function step(now) {
    const p = Math.min((now - start) / ms, 1);
    if (faceMaterial) faceMaterial.emissiveIntensity = from + (to - from) * p;
    if (p < 1) fadeAnimId = requestAnimationFrame(step);
    else { fadeAnimId = null; onDone?.(); }
  }
  fadeAnimId = requestAnimationFrame(step);
}
