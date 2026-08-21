// src/agent/rng.js
// Seeded PRNG (mulberry32) for the behavior brain.
// Same seed always produces the same sequence, so every brain
// decision is replayable in tests. No Math.random anywhere in the
// brain path.

export function mulberry32(seed) {
  let a = seed >>> 0;
  return function rng() {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Uniform float in [a, b). */
export function rangeFrom(rng, a, b) {
  return a + rng() * (b - a);
}

/** Weighted pick from [{value, weight}, ...]. Deterministic given rng. */
export function weightedPick(rng, entries) {
  let total = 0;
  for (const e of entries) total += e.weight;
  let roll = rng() * total;
  for (const e of entries) {
    roll -= e.weight;
    if (roll <= 0) return e.value;
  }
  return entries[entries.length - 1].value;
}
