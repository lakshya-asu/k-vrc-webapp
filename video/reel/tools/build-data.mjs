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

import {execFileSync} from 'node:child_process';
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
    '-af', 'loudnorm=I=-16:TP=-1.5:LRA=11',
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
