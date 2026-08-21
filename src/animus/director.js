import { validateActorPlan } from './contract.js';
import { deterministicActorPlan } from './fallback.js';

async function callWithTimeout(provider, request, timeoutMs) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(new Error('provider timeout')), timeoutMs);
  try {
    return await provider.plan(request, { signal: controller.signal });
  } finally {
    clearTimeout(timer);
  }
}

export class ActorDirector {
  constructor(options = {}) {
    this.providers = options.providers ?? [];
    this.fallback = options.fallback ?? deterministicActorPlan;
    this.timeoutMs = options.timeoutMs ?? 8000;
  }

  async plan(request) {
    const authority = {
      actorId: request.actorId ?? 'kvrc',
      controlLevel: request.controlLevel ?? 'suggest',
    };
    const failures = [];

    for (const provider of this.providers) {
      try {
        const candidate = await callWithTimeout(provider, request, this.timeoutMs);
        const checked = validateActorPlan(candidate, authority);
        if (checked.ok) {
          const provenance = {
            operator: provider.name,
            model: provider.model ?? null,
            fallback: false,
          };
          if (checked.value.beats.some((beat) => beat.face_glyph != null)) {
            // The model composed its own visor face; the receipt says
            // so explicitly (hand-authored glyphs are marked
            // "augmented" instead by the lanes that add them).
            provenance.face_glyph = 'model-authored';
          }
          return { ...checked.value, provenance };
        }
        failures.push({ provider: provider.name, reason: checked.errors.join('; ') });
      } catch (error) {
        failures.push({ provider: provider.name, reason: error?.message ?? String(error) });
      }
    }

    const fallbackCandidate = this.fallback(request);
    const checked = validateActorPlan(fallbackCandidate, authority);
    if (!checked.ok) {
      throw new Error(`deterministic fallback violated the actor contract: ${checked.errors.join('; ')}`);
    }

    return {
      ...checked.value,
      provenance: {
        operator: 'deterministic',
        model: null,
        fallback: true,
        prior_failures: failures,
      },
    };
  }
}
