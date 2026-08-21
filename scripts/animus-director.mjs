// Director runner: one script line in, one performed set of takes out.
//
//   script line -> actor plan (local model or deterministic fallback)
//               -> contract validation (src/animus/contract.js)
//               -> embodiment mapping (src/animus/embodiment.js)
//               -> voice pipeline for speech beats (Kokoro + Rhubarb)
//               -> typed bridge requests over the localhost socket
//
// Authority stays caller-owned: only --control-level perform opens a
// socket. suggest and preview map the plan and write the receipt
// without touching Blender.
//
// Examples:
//   node scripts/animus-director.mjs --line "Wave and say hello." \
//     --speech "Hello, I am K-VRC." --control-level perform --port 8765
//   node scripts/animus-director.mjs --fallback-only --voice-mode convert \
//     --emit-requests

import { readFileSync, writeFileSync, mkdirSync, existsSync } from 'node:fs';
import { spawnSync } from 'node:child_process';
import { createConnection } from 'node:net';
import path from 'node:path';
import process from 'node:process';
import { fileURLToPath } from 'node:url';

import { ActorDirector } from '../src/animus/director.js';
import { createOpenAICompatibleProvider } from '../src/animus/providers/openaiCompatible.js';
import {
  mapPlanToBridgeJobs,
  validateEmbodimentProfile,
  visemeArtifactToPoseRequest,
  visemeArtifactToRequest,
} from '../src/animus/embodiment.js';

const REPO = path.dirname(path.dirname(fileURLToPath(import.meta.url)));

function readArg(name, fallback = null) {
  const index = process.argv.indexOf(name);
  return index >= 0 ? process.argv[index + 1] : fallback;
}

const line = readArg('--line', 'Wave to the viewer and say hello.');
const speech = readArg('--speech', null);
const target = readArg('--target', 'camera');
const controlLevel = readArg('--control-level', 'suggest');
const profilePath = readArg(
  '--profile',
  path.join(REPO, 'src', 'animus', 'embodiment', 'kvrc-testrig.profile.json'),
);
const host = readArg('--host', '127.0.0.1');
const port = Number(readArg('--port', '8765'));
const voiceMode = readArg('--voice-mode', 'live'); // live | convert | skip
const receiptPath = readArg('--receipt', null);
const providerKind = readArg('--provider', 'local'); // local | stub-invalid
const fallbackOnly = process.argv.includes('--fallback-only');
const emitRequests = process.argv.includes('--emit-requests');

if (!['live', 'convert', 'skip'].includes(voiceMode)) {
  throw new Error(`unknown --voice-mode '${voiceMode}'`);
}

// The stub provider exists for the fallback acceptance path: it returns
// a plan that violates the contract (raw keyframes), so the validator
// must reject it and the deterministic fallback must perform instead.
function buildProviders() {
  if (fallbackOnly) return [];
  if (providerKind === 'stub-invalid') {
    return [{
      name: 'stub-invalid',
      model: 'stub',
      async plan() {
        return {
          schema_version: '0.1',
          summary: 'Invalid on purpose',
          keyframes: [{ bone: 'arm.R', frame: 1 }],
        };
      },
    }];
  }
  // A 4B model occasionally emits malformed JSON. Retrying is caller
  // policy, so the runner lists the same provider N times; the
  // director already walks providers in order and records every
  // failed attempt in provenance.prior_failures.
  const attempts = Math.max(1, Number(process.env.ANIMUS_LLM_ATTEMPTS ?? 3));
  return Array.from({ length: attempts }, (_, index) => createOpenAICompatibleProvider({
    baseUrl: process.env.ANIMUS_LLM_BASE_URL ?? 'http://127.0.0.1:8081/v1',
    model: process.env.ANIMUS_MODEL ?? 'Qwen3-4B-Q4_K_M',
    name: index === 0 ? 'local-small' : `local-small-retry${index}`,
  }));
}

function findVoicePython() {
  if (process.env.ANIMUS_VOICE_PYTHON) return process.env.ANIMUS_VOICE_PYTHON;
  const venv = path.join(REPO, '.venv-voice', 'Scripts', 'python.exe');
  if (existsSync(venv)) return venv;
  return 'python';
}

function runVoiceJob(job, outDir) {
  const python = findVoicePython();
  const common = [
    '--out', outDir,
    '--stem', job.stem,
    '--fps', String(job.fps),
    '--frame-start', String(job.frame_start),
    '--object', job.object,
    '--name-hint', job.name_hint,
  ];
  const args = voiceMode === 'live'
    ? ['-m', 'animus_voice', 'speak', '--text', job.text,
      '--voice', job.voice, '--lang', job.lang,
      '--speed', String(job.speed), '--seed', String(job.seed), ...common]
    : ['-m', 'animus_voice', 'convert',
      '--cues', path.join(REPO, 'voice', 'animus_voice', 'fixtures', 'hello.rhubarb.json'),
      ...common];

  const result = spawnSync(python, args, {
    cwd: REPO,
    env: { ...process.env, PYTHONPATH: path.join(REPO, 'voice') },
    encoding: 'utf-8',
    timeout: 300000,
  });
  if (result.status !== 0) {
    throw new Error(
      `voice pipeline failed for beat '${job.beat_id}' `
      + `(exit ${result.status}): ${result.stderr?.slice(-500)}`,
    );
  }
  const takePath = path.join(outDir, `${job.stem}.animus.json`);
  const artifact = JSON.parse(readFileSync(takePath, 'utf-8'));
  // Kokoro may print warnings to stdout before the tool's JSON report;
  // the artifact file above is the authoritative output either way.
  let toolReport = null;
  const start = result.stdout.indexOf('{');
  const end = result.stdout.lastIndexOf('}');
  if (start >= 0 && end > start) {
    try {
      toolReport = JSON.parse(result.stdout.slice(start, end + 1));
    } catch {
      toolReport = null;
    }
  }
  return { artifact, takePath, tool_report: toolReport };
}

function sendRequest(request) {
  return new Promise((resolve, reject) => {
    const socket = createConnection({ host, port }, () => {
      socket.write(`${JSON.stringify(request)}\n`);
    });
    let buffer = '';
    socket.setTimeout(60000, () => {
      socket.destroy();
      reject(new Error(`bridge did not answer request '${request.id}' within 60s`));
    });
    socket.on('data', (chunk) => {
      buffer += chunk.toString('utf-8');
      const newline = buffer.indexOf('\n');
      if (newline >= 0) {
        socket.end();
        try {
          resolve(JSON.parse(buffer.slice(0, newline)));
        } catch (error) {
          reject(error);
        }
      }
    });
    socket.on('error', reject);
  });
}

const profileCheck = validateEmbodimentProfile(
  JSON.parse(readFileSync(profilePath, 'utf-8')),
);
if (!profileCheck.ok) {
  process.stderr.write(`embodiment profile is invalid:\n${profileCheck.errors.join('\n')}\n`);
  process.exit(2);
}
const profile = profileCheck.value;

const director = new ActorDirector({
  providers: buildProviders(),
  timeoutMs: Number(process.env.ANIMUS_TIMEOUT_MS ?? 30000),
});
const plan = await director.plan({
  actorId: process.env.ANIMUS_ACTOR_ID ?? 'kvrc',
  controlLevel,
  instruction: line,
  target,
  speech,
  capabilities: {
    body: true, gaze: true, face: true, face_glyph: true, speech: true,
  },
});

const jobs = mapPlanToBridgeJobs(plan, profile);

// Speech beats become bridge requests only after the voice pipeline
// runs. skip mode maps and performs everything except speech.
const voiceReceipts = [];
if (voiceMode !== 'skip') {
  const outDir = path.join(REPO, 'voice', 'animus_voice', 'out', 'director');
  mkdirSync(outDir, { recursive: true });
  for (const job of jobs.voice_jobs) {
    const { artifact, takePath, tool_report } = runVoiceJob(job, outDir);
    const speechRequest = profile.speech.mode === 'bone'
      ? visemeArtifactToPoseRequest(artifact, profile.speech, {
        requestId: `dir-${job.stem}`,
        object: job.object,
      })
      : visemeArtifactToRequest(artifact, {
        requestId: `dir-${job.stem}`,
        object: job.object,
      });
    jobs.layers.push({
      beat_id: job.beat_id,
      channel: 'speech',
      request: speechRequest,
    });
    voiceReceipts.push({
      beat_id: job.beat_id,
      take_json: takePath,
      voice_mode: voiceMode,
      tool_report,
    });
  }
}

if (emitRequests) {
  process.stdout.write(`${JSON.stringify({
    profile: jobs.profile,
    requests: jobs.layers.map((layer) => layer.request),
  }, null, 2)}\n`);
}

// Only perform opens a socket. The plan's control_level came back from
// the validator, so it is the caller's value, never the model's.
const performed = plan.control_level === 'perform';
if (performed) {
  for (const layer of jobs.layers) {
    layer.response = await sendRequest(layer.request);
  }
}

const receipt = {
  kind: 'animus_director_receipt',
  line,
  requested_speech: speech,
  control_level: plan.control_level,
  performed,
  voice_mode: voiceMode,
  plan,
  profile: jobs.profile,
  layers: jobs.layers,
  voice: voiceReceipts,
  bridge: performed ? { host, port } : null,
};
if (receiptPath) {
  mkdirSync(path.dirname(path.resolve(receiptPath)), { recursive: true });
  writeFileSync(receiptPath, `${JSON.stringify(receipt, null, 2)}\n`);
}

const failures = jobs.layers.filter((layer) => performed && layer.response?.ok !== true);
if (!emitRequests) {
  process.stdout.write(`${JSON.stringify({
    control_level: plan.control_level,
    performed,
    operator: plan.provenance.operator,
    fallback: plan.provenance.fallback,
    beats: plan.beats.length,
    layers: jobs.layers.map((layer) => ({
      beat_id: layer.beat_id,
      channel: layer.channel,
      request_id: layer.request.id,
      ok: performed ? layer.response?.ok === true : null,
      receipt: performed ? layer.response?.result ?? layer.response?.error : null,
    })),
    receipt_path: receiptPath,
  }, null, 2)}\n`);
}
// exitCode instead of process.exit(): a hard exit can abort libuv on
// Windows while spawned-process handles are still closing.
process.exitCode = failures.length > 0 ? 1 : 0;
