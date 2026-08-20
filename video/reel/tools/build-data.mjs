#!/usr/bin/env node
// Prepare the reel's inputs from a rendered scene batch:
//
//   node tools/build-data.mjs --scenes <dir with scene-01..NN/scene.mp4> \
//        --manifest <scene batch manifest.json>
//
// For every scene: loudness-normalize the audio track (ffmpeg loudnorm,
// -16 LUFS / -1.5 dBTP, video stream copied untouched) into
// public/scenes/scene-NN.mp4, probe the exact duration, and write
// src/reel-data.json with the caption metadata the composition uses.
// Requires ffmpeg + ffprobe on PATH.
//
// Loudnorm runs TWO passes: a measurement pass, then a linear
// (constant-gain) pass using the measured values. Single-pass loudnorm
// is dynamic: it rides the gain over time, which pumps the noise floor
// up inside the silent head/tail room around each line. Linear gain
// keeps silence silent, which the reel's dialogue gaps depend on.

import {execFileSync, spawnSync} from 'node:child_process';
import {existsSync, mkdirSync, readFileSync, writeFileSync} from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.dirname(HERE);

function parseArgs(argv) {
  const args = {};
  for (let i = 0; i < argv.length; i++) {
    if (argv[i] === '--scenes') args.scenes = argv[++i];
    else if (argv[i] === '--manifest') args.manifest = argv[++i];
  }
  if (!args.scenes || !args.manifest) {
    console.error('usage: build-data.mjs --scenes <dir> --manifest <json>');
    process.exit(2);
  }
  return args;
}

const LOUDNORM = 'loudnorm=I=-16:TP=-1.5:LRA=11';

// Measurement pass: ffmpeg prints the loudnorm stats as a JSON block on
// stderr. Returns the linear second-pass filter string, or the plain
// dynamic filter if measurement fails (it should not on a real scene).
function measuredFilter(file) {
  const run = spawnSync('ffmpeg', [
    '-hide_banner', '-nostats', '-i', file,
    '-af', `${LOUDNORM}:print_format=json`,
    '-f', 'null', '-',
  ], {stdio: ['ignore', 'ignore', 'pipe']});
  const text = (run.stderr ?? '').toString();
  const match = text.match(/\{[\s\S]*?"input_i"[\s\S]*?\}/);
  if (!match) {
    console.warn(`loudnorm measurement failed for ${file}; using dynamic pass`);
    return LOUDNORM;
  }
  const m = JSON.parse(match[0]);
  return (
    `${LOUDNORM}:measured_I=${m.input_i}:measured_TP=${m.input_tp}` +
    `:measured_LRA=${m.input_lra}:measured_thresh=${m.input_thresh}` +
    `:offset=${m.target_offset}:linear=true`
  );
}

function probeDuration(file) {
  const out = execFileSync('ffprobe', [
    '-v', 'error',
    '-show_entries', 'format=duration',
    '-of', 'csv=p=0',
    file,
  ]).toString().trim();
  return Number(out);
}

function shortDescription(sceneEntry) {
  // First sentence of the stage direction, without trailing period.
  const direction = sceneEntry.stage_direction || '';
  const first = direction.split(/(?<=\.)\s/)[0].replace(/\.$/, '');
  return first;
}

const args = parseArgs(process.argv.slice(2));
const manifest = JSON.parse(readFileSync(args.manifest, 'utf-8'));
const outDir = path.join(ROOT, 'public', 'scenes');
mkdirSync(outDir, {recursive: true});

const scenes = [];
for (const entry of manifest.scenes) {
  const num = entry.scene;
  const id = `scene-${String(num).padStart(2, '0')}`;
  const src = path.join(args.scenes, id, 'scene.mp4');
  if (!existsSync(src)) {
    console.error(`missing ${src}`);
    process.exit(1);
  }
  const dst = path.join(outDir, `${id}.mp4`);
  execFileSync('ffmpeg', [
    '-y', '-i', src,
    '-af', measuredFilter(src),
    '-c:v', 'copy',
    '-c:a', 'aac', '-b:a', '192k',
    dst,
  ], {stdio: ['ignore', 'ignore', 'pipe']});
  const durationS = probeDuration(dst);
  scenes.push({
    num,
    name: entry.name,
    description: shortDescription(entry),
    file: `scenes/${id}.mp4`,
    durationS,
  });
  console.log(`${id}: ${durationS.toFixed(2)}s normalized -> ${dst}`);
}

const data = {generated: new Date().toISOString(), scenes};
writeFileSync(
  path.join(ROOT, 'src', 'reel-data.json'),
  JSON.stringify(data, null, 1) + '\n',
);
console.log(`wrote src/reel-data.json with ${scenes.length} scenes`);
