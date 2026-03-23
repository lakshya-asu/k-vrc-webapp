import * as THREE from 'three';

// All six emotion states and their scene effects.
// "thinking" is client-side only — never returned by Gemini.
// headJerk: triggers a rapid head-shake (see robot.js triggerHeadJerk)
// headRapid: multiplies head lerp speed for rapid tracking
const EMOTION_MAP = {
  neutral:  { rimHex: 0x00e5ff, bloom: 0.9,  floatSpeed: 0.5,  floatAmp: 0.05, headJerk: false, headRapid: false, bodyMotion: 'sway'  },
  happy:    { rimHex: 0x00ff88, bloom: 1.2,  floatSpeed: 0.9,  floatAmp: 0.07, headJerk: false, headRapid: false, bodyMotion: 'nod'   },
  excited:  { rimHex: 0xffe600, bloom: 1.8,  floatSpeed: 1.4,  floatAmp: 0.05, headJerk: false, headRapid: true,  bodyMotion: 'sway'  },
  sad:      { rimHex: 0x4488ff, bloom: 0.4,  floatSpeed: 0.3,  floatAmp: 0.03, headJerk: false, headRapid: false, bodyMotion: 'slump' },
  angry:    { rimHex: 0xff2200, bloom: 1.6,  floatSpeed: 1.2,  floatAmp: 0.06, headJerk: true,  headRapid: false, bodyMotion: 'shake' },
  thinking: { rimHex: 0xbb44ff, bloom: 0.6,  floatSpeed: 0.4,  floatAmp: 0.04, headJerk: false, headRapid: false, bodyMotion: 'tilt'  },
};

const _rimColor = new THREE.Color();

/**
 * Apply emotion effects to scene lights and bloom pass.
 * @param {string} emotion - one of the keys in EMOTION_MAP
 * @param {{ rimLight, faceLight, bloomPass }} refs
 */
export function applyEmotion(emotion, { rimLight, faceLight, bloomPass }) {
  const cfg = EMOTION_MAP[emotion] ?? EMOTION_MAP.neutral;
  _rimColor.setHex(cfg.rimHex);
  rimLight.color.copy(_rimColor);
  faceLight.color.copy(_rimColor);
  bloomPass.strength = cfg.bloom;
  return cfg; // caller can use floatSpeed / floatAmp / headJerk / headRapid
}

export { EMOTION_MAP };
