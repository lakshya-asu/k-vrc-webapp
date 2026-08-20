#!/usr/bin/env node
// Render the K-VRC visor face for one take as a PNG frame sequence.
//
//   node scripts/render-face-frames.mjs --receipt <take.json> --out <dir>
//        [--seed 0] [--fps N] [--frames A-B]
//
// The input is either a full actor receipt (kind animus_actor_receipt,
// what --receipt from python -m animus_actor writes) or the compact
// face job the actor loop prepares (kind animus_face_job). Both reduce
// to the same job: fps, frame range, face beats, viseme samples, seed.
//
// Output: face_0001.png .. face_NNNN.png (512x512, the webapp's face
// canvas) plus face-frames-manifest.json with the seed, range, and a
// sha256 per frame so a rerun can be verified byte for byte.
//
// Drawing runs the ported webapp renderer (src/animus/face/) under
// @napi-rs/canvas. Deterministic: same input and seed, same bytes.

import { createHash } from 'node:crypto';
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import path from 'node:path';

import { EXPRESSION_LIBRARY } from '../src/animus/face/expressionLibrary.js';
import { drawFaceFrame, W, H } from '../src/animus/face/faceScreenDraw.js';
import { buildFaceTimeline, mulberry32 } from '../src/animus/face/faceTimeline.js';

const require = createRequire(import.meta.url);
const { createCanvas } = require('@napi-rs/canvas');

function parseArgs(argv) {
  const args = { seed: 0, fps: null, frames: null };
  for (let i = 0; i < argv.length; i++) {
    const key = argv[i];
    const next = () => argv[++i];
    if (key === '--receipt') args.receipt = next();
    else if (key === '--out') args.out = next();
    else if (key === '--seed') args.seed = Number(next());
    else if (key === '--fps') args.fps = Number(next());
    else if (key === '--frames') args.frames = next();
    else throw new Error(`unknown argument: ${key}`);
  }
  if (!args.receipt || !args.out) {
    throw new Error('usage: render-face-frames.mjs --receipt <take.json> --out <dir> [--seed N] [--fps N] [--frames A-B]');
  }
  return args;
}

function loadJson(file) {
  return JSON.parse(readFileSync(file, 'utf-8'));
}

// A full actor receipt reduces to the compact face job.
function jobFromReceipt(receipt, receiptPath, fpsOverride) {
  if (receipt.kind === 'animus_face_job') return receipt;
  if (receipt.kind !== 'animus_actor_receipt') {
    throw new Error(`unsupported input kind '${receipt.kind}'`);
  }
  const beats = receipt.plan?.beats || [];
  const faceBeats = beats
    .filter((beat) => beat.face)
    .map((beat) => ({
      expression: beat.face.expression,
      intensity: beat.face.intensity ?? 1,
      at_ms: beat.at_ms,
      duration_ms: beat.duration_ms,
    }));

  const visemeSamples = [];
  let fps = fpsOverride;
  let frameEnd = 1;
  for (const voice of receipt.voice || []) {
    const takeJson = voice.take_json;
    if (!takeJson) continue;
    const takePath = path.isAbsolute(takeJson)
      ? takeJson
      : path.join(path.dirname(receiptPath), takeJson);
    const artifact = loadJson(takePath);
    visemeSamples.push(...(artifact.samples || []));
    fps = fps || artifact.fps;
    frameEnd = Math.max(frameEnd, artifact.frame_end || 1);
  }
  for (const layer of receipt.layers || []) {
    const end = layer.request?.params?.frame_end;
    if (Number.isInteger(end)) frameEnd = Math.max(frameEnd, end);
  }
  return {
    kind: 'animus_face_job',
    fps: fps || 24,
    frame_end: frameEnd,
    face_beats: faceBeats,
    viseme_samples: visemeSamples,
    seed: 0,
  };
}

function main() {
  const args = parseArgs(process.argv.slice(2));
  const receipt = loadJson(args.receipt);
  const job = jobFromReceipt(receipt, path.resolve(args.receipt), args.fps);
  if (Number.isFinite(args.seed)) job.seed = args.seed >>> 0;
  if (args.fps) job.fps = args.fps;

  const timeline = buildFaceTimeline(job, EXPRESSION_LIBRARY);

  let lo = 1;
  let hi = job.frame_end;
  if (args.frames) {
    const [a, b] = args.frames.split('-').map(Number);
    lo = Math.max(1, a);
    hi = Math.min(job.frame_end, b || a);
  }

  mkdirSync(args.out, { recursive: true });
  const canvas = createCanvas(W, H);
  const ctx = canvas.getContext('2d');

  // One draw rng per frame, seeded from the job seed and the frame
  // number, so rendering a sub-range produces the same bytes as the
  // same frames from a full render.
  const manifest = {
    kind: 'animus_face_frames',
    seed: job.seed,
    fps: job.fps,
    frame_start: lo,
    frame_end: hi,
    frame_count: hi - lo + 1,
    size: [W, H],
    face_beats: job.face_beats,
    frames: [],
  };
  for (const state of timeline) {
    if (state.frame < lo || state.frame > hi) continue;
    const drawRng = mulberry32(((job.seed >>> 0) ^ (state.frame * 0x9e3779b9)) >>> 0);
    drawFaceFrame(ctx, state, drawRng);
    const png = canvas.toBuffer('image/png');
    const name = `face_${String(state.frame).padStart(4, '0')}.png`;
    writeFileSync(path.join(args.out, name), png);
    manifest.frames.push({
      frame: state.frame,
      file: name,
      mood: state.mood,
      amplitude: Number(state.amplitude.toFixed(6)),
      blink: Number(state.blinkProgress.toFixed(6)),
      sha256: createHash('sha256').update(png).digest('hex'),
    });
  }
  writeFileSync(
    path.join(args.out, 'face-frames-manifest.json'),
    JSON.stringify(manifest, null, 2) + '\n',
  );
  console.log(JSON.stringify({
    ok: true,
    out: args.out,
    frame_count: manifest.frame_count,
    seed: job.seed,
    fps: job.fps,
    moods: [...new Set(manifest.frames.map((f) => f.mood))],
  }));
}

main();
