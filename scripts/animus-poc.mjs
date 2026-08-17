import { ActorDirector } from '../src/animus/director.js';
import { createOpenAICompatibleProvider } from '../src/animus/providers/openaiCompatible.js';

function readArg(name, fallback = null) {
  const index = process.argv.indexOf(name);
  return index >= 0 ? process.argv[index + 1] : fallback;
}

const fallbackOnly = process.argv.includes('--fallback-only');
const instruction = readArg('--instruction', 'Wave to the viewer and say hello.');
const target = readArg('--target', 'camera');
const speech = readArg('--speech', null);
const controlLevel = readArg('--control-level', 'suggest');

const providers = fallbackOnly
  ? []
  : [createOpenAICompatibleProvider({
      baseUrl: process.env.ANIMUS_LLM_BASE_URL ?? 'http://127.0.0.1:8081/v1',
      model: process.env.ANIMUS_MODEL ?? 'Qwen3-4B-Q4_K_M',
    })];

const timeoutMs = Number(process.env.ANIMUS_TIMEOUT_MS ?? 8000);
const director = new ActorDirector({ providers, timeoutMs });
const plan = await director.plan({
  actorId: process.env.ANIMUS_ACTOR_ID ?? 'kvrc',
  controlLevel,
  instruction,
  target,
  speech,
  capabilities: {
    body: true,
    gaze: true,
    face: true,
    speech: true,
  },
});

process.stdout.write(`${JSON.stringify(plan, null, 2)}\n`);
