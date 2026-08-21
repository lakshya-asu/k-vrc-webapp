// src/agent/glyphComposer.js
// Generative glyph face composer, ported from the animus repo
// (src/animus/face/glyphComposer.js on feat/animus-poc).
//
// K-VRC's faces are COMPOSED GLYPHS, not only fixed library entries.
// A face is eyes + brows + mouth picked from a small parametric
// vocabulary, or a short piece of LED text filling the screen. The
// behavior brain emits these through 'glyph' actions; everything here
// is strictly validated so no caller can push arbitrary drawing.
//
// This module is dependency-free on purpose: no three.js, no DOM
// globals. It draws onto whatever 2d context faceScreen hands it,
// using the same geometry constants, so composed faces sit exactly
// where library faces sit (the blink eyelid overlay and speech bob
// keep working).

// Mood names must match MOOD_COLORS in src/faceScreen.js. Kept as a
// plain list here so this module stays importable in Node tests
// without pulling in three.js.
export const GLYPH_MOODS = [
  'cold', 'warm', 'glitch', 'static', 'data', 'boot', 'angry', 'dream',
];

export const GLYPH_EYES = [
  'round',
  'oval',
  'bar',
  'happy_arc',
  'half_lidded',
  'closed',
  'wide',
  'x_cross',
];

export const GLYPH_BROWS = [
  'none',
  'flat',
  'raised',
  'angry_in',
  'sad_out',
];

export const GLYPH_MOUTHS = [
  'none',
  'flat',
  'smile',
  'frown',
  'o_small',
  'grin_rect',
  'gritted',
  'v_smile',
  'wavy',
];

export const GLYPH_TEXT_MAX = 6;
// LED marquee charset: uppercase letters, digits, basic punctuation.
export const GLYPH_TEXT_PATTERN = /^[A-Z0-9 !?%+\-*#<>:=._]+$/;

const SPEC_KEYS = new Set(['eyes', 'brows', 'mouth', 'text', 'mood', 'intensity']);

function isObject(value) {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

// Strict validation. Returns { ok, errors, value }; value is the
// normalized spec (text uppercased and trimmed, defaults filled).
export function validateFaceGlyph(spec, path = 'face_glyph') {
  const errors = [];
  if (!isObject(spec)) {
    return { ok: false, errors: [`${path} must be an object`], value: null };
  }
  for (const key of Object.keys(spec)) {
    if (!SPEC_KEYS.has(key)) errors.push(`${path}.${key} is not allowed`);
  }

  const hasText = spec.text !== undefined && spec.text !== null;
  const value = {};

  if (hasText) {
    for (const key of ['eyes', 'brows', 'mouth']) {
      if (spec[key] !== undefined) {
        errors.push(`${path}.${key} is not allowed in text mode`);
      }
    }
    if (typeof spec.text !== 'string') {
      errors.push(`${path}.text must be a string`);
    } else {
      const text = spec.text.trim().toUpperCase();
      if (text.length < 1 || text.length > GLYPH_TEXT_MAX) {
        errors.push(`${path}.text must be 1 to ${GLYPH_TEXT_MAX} characters`);
      } else if (!GLYPH_TEXT_PATTERN.test(text)) {
        errors.push(`${path}.text may use only A-Z 0-9 and ! ? % + - * # < > : = . _`);
      } else {
        value.text = text;
      }
    }
  } else {
    if (!GLYPH_EYES.includes(spec.eyes)) {
      errors.push(`${path}.eyes must be one of: ${GLYPH_EYES.join(', ')}`);
    } else {
      value.eyes = spec.eyes;
    }
    const brows = spec.brows ?? 'none';
    if (!GLYPH_BROWS.includes(brows)) {
      errors.push(`${path}.brows must be one of: ${GLYPH_BROWS.join(', ')}`);
    } else {
      value.brows = brows;
    }
    const mouth = spec.mouth ?? 'none';
    if (!GLYPH_MOUTHS.includes(mouth)) {
      errors.push(`${path}.mouth must be one of: ${GLYPH_MOUTHS.join(', ')}`);
    } else {
      value.mouth = mouth;
    }
  }

  if (spec.mood !== undefined) {
    if (!GLYPH_MOODS.includes(spec.mood)) {
      errors.push(`${path}.mood must be one of: ${GLYPH_MOODS.join(', ')}`);
    } else {
      value.mood = spec.mood;
    }
  }
  if (spec.intensity !== undefined) {
    const i = spec.intensity;
    if (typeof i !== 'number' || !Number.isFinite(i) || i < 0 || i > 1) {
      errors.push(`${path}.intensity must be a number from 0 to 1`);
    } else {
      value.intensity = i;
    }
  }

  if (errors.length > 0) return { ok: false, errors, value: null };
  if (value.intensity === undefined) value.intensity = 1;
  return { ok: true, errors: [], value };
}

// --- drawing --------------------------------------------------------------
// Shared geometry: identical eye/mouth anchors to drawWeighted in
// faceScreen.js so the blink overlay and speech scale bob line up.
const W = 512;
const H = 512;
const EY = H * 0.42;
const EX_OFF = W * 0.175;
const ES = W * 0.12;
const MY = H * 0.65;

function drawEye(ctx, kind, ex, blink, k) {
  const open = Math.max(1 - blink, 0.02);
  ctx.lineWidth = 9 * k;
  ctx.lineCap = 'round';
  switch (kind) {
    case 'round':
      ctx.beginPath();
      ctx.ellipse(ex, EY, ES * 0.52 * k, Math.max(ES * 0.62 * k * open, 2), 0, 0, Math.PI * 2);
      ctx.fill();
      break;
    case 'oval':
      ctx.beginPath();
      ctx.ellipse(ex, EY, ES * 0.42 * k, Math.max(ES * 0.8 * k * open, 2), 0, 0, Math.PI * 2);
      ctx.fill();
      break;
    case 'wide':
      ctx.beginPath();
      ctx.ellipse(ex, EY, ES * 0.68 * k, Math.max(ES * 0.72 * k * open, 2), 0, 0, Math.PI * 2);
      ctx.fill();
      break;
    case 'bar':
      ctx.fillRect(ex - ES * 0.55 * k, EY - ES * 0.16 * k, ES * 1.1 * k, ES * 0.32 * k);
      break;
    case 'happy_arc':
      // ^ ^ closed happy eyes.
      ctx.beginPath();
      ctx.arc(ex, EY + ES * 0.35, ES * 0.55 * k, Math.PI * 1.15, Math.PI * 1.85);
      ctx.stroke();
      break;
    case 'half_lidded': {
      // Bored: the top half of the eye is cut flat.
      const h = Math.max(ES * 0.62 * k * open, 2);
      ctx.save();
      ctx.beginPath();
      ctx.rect(ex - ES, EY - h * 0.15, ES * 2, h * 1.2);
      ctx.clip();
      ctx.beginPath();
      ctx.ellipse(ex, EY, ES * 0.52 * k, h, 0, 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();
      ctx.fillRect(ex - ES * 0.55 * k, EY - h * 0.24, ES * 1.1 * k, 4);
      break;
    }
    case 'closed':
      ctx.beginPath();
      ctx.moveTo(ex - ES * 0.5 * k, EY);
      ctx.lineTo(ex + ES * 0.5 * k, EY);
      ctx.stroke();
      break;
    case 'x_cross':
      ctx.beginPath();
      ctx.moveTo(ex - ES * 0.4 * k, EY - ES * 0.4 * k);
      ctx.lineTo(ex + ES * 0.4 * k, EY + ES * 0.4 * k);
      ctx.moveTo(ex + ES * 0.4 * k, EY - ES * 0.4 * k);
      ctx.lineTo(ex - ES * 0.4 * k, EY + ES * 0.4 * k);
      ctx.stroke();
      break;
    default:
      break;
  }
}

function drawBrows(ctx, kind, k) {
  if (kind === 'none') return;
  const browY = EY - ES * 1.05;
  ctx.lineWidth = 9 * k;
  ctx.lineCap = 'round';
  const tilt = ES * 0.42 * k;
  for (const side of [-1, 1]) {
    const ex = W / 2 + side * EX_OFF;
    ctx.beginPath();
    switch (kind) {
      case 'flat':
        ctx.moveTo(ex - ES * 0.5, browY);
        ctx.lineTo(ex + ES * 0.5, browY);
        break;
      case 'raised':
        ctx.moveTo(ex - ES * 0.5, browY - ES * 0.35 * k);
        ctx.lineTo(ex + ES * 0.5, browY - ES * 0.35 * k);
        break;
      case 'angry_in':
        // Inner ends slant down toward the nose.
        ctx.moveTo(ex - side * ES * 0.55, browY - tilt * 0.55);
        ctx.lineTo(ex + side * ES * 0.45, browY + tilt * 0.55);
        break;
      case 'sad_out':
        ctx.moveTo(ex - side * ES * 0.55, browY + tilt * 0.5);
        ctx.lineTo(ex + side * ES * 0.45, browY - tilt * 0.5);
        break;
      default:
        break;
    }
    ctx.stroke();
  }
}

function roundedRect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}

function drawMouth(ctx, kind, k, amplitude, c) {
  if (kind === 'none') return;
  ctx.lineWidth = 8 * k;
  ctx.lineCap = 'round';
  const openBoost = 1 + amplitude * 0.55;
  switch (kind) {
    case 'flat':
      ctx.beginPath();
      ctx.moveTo(W / 2 - 26 * k, MY);
      ctx.lineTo(W / 2 + 26 * k, MY);
      ctx.stroke();
      break;
    case 'smile':
      ctx.beginPath();
      ctx.arc(W / 2, MY - 12, (26 + 26 * k), 0.12 * Math.PI, 0.88 * Math.PI);
      ctx.stroke();
      break;
    case 'frown':
      ctx.beginPath();
      ctx.arc(W / 2, MY + 34, (24 + 24 * k), 1.15 * Math.PI, 1.85 * Math.PI);
      ctx.stroke();
      break;
    case 'v_smile':
      ctx.beginPath();
      ctx.moveTo(W / 2 - 20 * k, MY - 8 * k);
      ctx.lineTo(W / 2, MY + 10 * k);
      ctx.lineTo(W / 2 + 20 * k, MY - 8 * k);
      ctx.stroke();
      break;
    case 'o_small':
      ctx.beginPath();
      ctx.ellipse(W / 2, MY, 16 * k, 18 * k * openBoost, 0, 0, Math.PI * 2);
      ctx.fill();
      break;
    case 'grin_rect': {
      // Big open grin: large rounded-square mouth.
      const gw = 150 * k;
      const gh = 84 * k * openBoost;
      roundedRect(ctx, W / 2 - gw / 2, MY - gh / 2, gw, gh, 20 * k);
      ctx.fill();
      break;
    }
    case 'gritted': {
      // Gritted flat mouth with teeth separators.
      const gw = 160 * k;
      const gh = 44 * k;
      roundedRect(ctx, W / 2 - gw / 2, MY - gh / 2, gw, gh, 10 * k);
      ctx.fill();
      ctx.save();
      ctx.strokeStyle = c.bg;
      ctx.lineWidth = 6 * k;
      for (let i = 1; i <= 3; i++) {
        const x = W / 2 - gw / 2 + (gw * i) / 4;
        ctx.beginPath();
        ctx.moveTo(x, MY - gh / 2 + 4);
        ctx.lineTo(x, MY + gh / 2 - 4);
        ctx.stroke();
      }
      ctx.restore();
      break;
    }
    case 'wavy': {
      ctx.beginPath();
      const span = 60 * k;
      ctx.moveTo(W / 2 - span, MY);
      for (let i = 1; i <= 4; i++) {
        const x = W / 2 - span + (span * 2 * i) / 4;
        const y = MY + (i % 2 === 0 ? 10 : -10) * k;
        ctx.quadraticCurveTo(
          W / 2 - span + (span * 2 * (i - 0.5)) / 4,
          y,
          x,
          MY,
        );
      }
      ctx.stroke();
      break;
    }
    default:
      break;
  }
}

function drawText(ctx, text, k) {
  const size = Math.min(220, (W * 0.94) / (text.length * 0.72));
  ctx.font = `900 ${Math.round(size * (0.8 + 0.2 * k))}px "Arial Black", Arial, sans-serif`;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText(text, W / 2, H / 2);
}

// One composed glyph face. spec must have passed validateFaceGlyph.
// c: the mood palette {primary, secondary, bg}; blink 0..1;
// amplitude 0..1 speech loudness.
export function drawGlyphFace(ctx, spec, c, blink, amplitude) {
  const k = 0.65 + 0.35 * (spec.intensity ?? 1);
  ctx.fillStyle = c.primary;
  ctx.strokeStyle = c.primary;
  if (spec.text) {
    drawText(ctx, spec.text, k);
    return;
  }
  for (const side of [-1, 1]) {
    drawEye(ctx, spec.eyes, W / 2 + side * EX_OFF, blink, k);
  }
  drawBrows(ctx, spec.brows, k);
  drawMouth(ctx, spec.mouth, k, amplitude, c);
}
