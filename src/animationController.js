import * as THREE from 'three';

// Maps LLM gesture names → one or more clip name candidates (tries in order)
// Clip names must match NLA strip names from Blender export
const GESTURE_CLIPS = {
  // Idle/passive
  idle:            ['breathing_idle', 'happy_idle', 'weight_shift'],
  think:           ['thinking', 'focus', 'weight_shift'],
  listen:          ['acknowledging', 'lengthy_head_nod', 'head_nod_yes'],

  // Positive
  happy:           ['happy_idle', 'happy_hand_gesture', 'reacting'],
  excited:         ['victory_idle', 'rallying', 'reacting'],
  laugh:           ['laughing', 'reacting', 'victory_idle'],
  wave:            ['waving', 'waving_1'],
  celebrate:       ['victory_idle', 'victory_idle_1', 'rallying'],
  thankful:        ['thankful', 'acknowledging'],
  dance:           ['hip_hop_dancing', 'silly_dancing', 'samba_dancing', 'bestdance'],

  // Communicative
  talk:            ['talking', 'explaining', 'telling_a_secret'],
  explain:         ['explaining', 'talking'],
  secret:          ['telling_a_secret', 'talking'],

  // Negative / reactive
  sad:             ['sad_idle', 'sad_idle_1', 'rejected'],
  angry:           ['angry', 'standing_arguing', 'angry_gesture'],
  dismiss:         ['dismissing_gesture', 'look_away_gesture', 'being_cocky'],
  reject:          ['rejected', 'sad_idle', 'shaking_head_no'],
  shrug:           ['reacting', 'look_away_gesture'],

  // Head gestures
  nod:             ['head_nod_yes', 'hard_head_nod', 'lengthy_head_nod'],
  nod_sarcastic:   ['sarcastic_head_nod', 'annoyed_head_shake'],
  shake_no:        ['shaking_head_no', 'annoyed_head_shake'],
  sigh:            ['relieved_sigh', 'sad_idle'],
  cocky:           ['being_cocky', 'taunt_gesture'],
};

// Which gestures should loop vs play once
const LOOPING = new Set(['idle', 'think', 'listen', 'dance', 'talk', 'happy', 'excited', 'sad', 'angry', 'cocky']);

// All idle-category clips to rotate through randomly
// Mix of full-body dances, expressive idles, and character moments
const IDLE_POOL = [
  'breathing_idle',
  'happy_idle',
  'weight_shift',
  'hip_hop_dancing',
  'silly_dancing',
  'samba_dancing',
  'being_cocky',
  'victory_idle',
  'thankful',
  'rallying',
  'reacting',
  'relieved_sigh',
  'look_away_gesture',
  'sad_idle',
];

export class AnimationController {
  constructor() {
    this.mixer = null;
    this.clips = {};
    this.current = null;
    this.currentName = null;
    this.fadeTime = 0.4;
    this._idleTimer = 0;
    this._idleInterval = 6 + Math.random() * 6; // 6-12s — long enough for dances to play out
    this._lastIdleClip = null;
  }

  // Normalize a clip name for fuzzy matching:
  // "Breathing Idle"     → "breathing_idle"
  // "Hip Hop Dancing"    → "hip_hop_dancing"
  // "breathing_idle.002" → "breathing_idle"   ← Blender duplicate suffix
  static normalize(s) {
    return s.toLowerCase().trim()
      .replace(/\.\d+$/, '')          // strip Blender .002/.003 suffixes
      .replace(/[\s\-]+/g, '_')       // spaces/dashes → underscore
      .replace(/[^a-z0-9_]/g, '');   // drop remaining special chars
  }

  /** Call after GLB is loaded. Pass gltf.animations array. */
  init(root, animations) {
    this.mixer = new THREE.AnimationMixer(root);

    for (const clip of animations) {
      this.clips[clip.name] = clip;
      const norm = AnimationController.normalize(clip.name);
      // Only store normalized form if not already claimed (first occurrence wins)
      if (!this.clips[norm]) this.clips[norm] = clip;
    }

    console.log(`AnimationController: ${animations.length} clips loaded:`,
      animations.map(a => a.name).sort().join(', '));
  }

  /** Play a gesture by LLM gesture name. Returns true if clip found. */
  playGesture(gestureName) {
    // Idle gets random variety
    if (gestureName === 'idle') {
      this._playRandomIdle();
      return true;
    }

    const candidates = GESTURE_CLIPS[gestureName];
    if (!candidates) return false;

    // Pick randomly from available candidates instead of always first
    const available = candidates.filter(c => this.clips[c]);
    if (!available.length) {
      console.warn(`AnimationController: no clip for gesture "${gestureName}". Tried:`, candidates);
      return false;
    }
    const pick = available[Math.floor(Math.random() * available.length)];
    return this._play(pick, LOOPING.has(gestureName));
  }

  /** Play a clip directly by name. */
  playClip(clipName, loop = false) {
    return this._play(clipName, loop);
  }

  _play(clipName, loop = false) {
    if (this.currentName === clipName) return true;

    const clip = this.clips[clipName];
    if (!clip) return false;

    const next = this.mixer.clipAction(clip);
    next.setLoop(loop ? THREE.LoopRepeat : THREE.LoopOnce);
    next.clampWhenFinished = !loop;
    next.reset();

    if (this.current && this.current !== next) {
      this.current.crossFadeTo(next, this.fadeTime, true);
    }
    next.play();

    this.current = next;
    this.currentName = clipName;

    // Auto-return to idle when one-shot finishes
    if (!loop) {
      const onFinish = (e) => {
        if (e.action === next) {
          this.mixer.removeEventListener('finished', onFinish);
          this.currentName = null;
          this.playGesture('idle');
        }
      };
      this.mixer.addEventListener('finished', onFinish);
    }

    return true;
  }

  /** Must be called every frame with delta seconds. */
  update(delta) {
    if (!this.mixer) return;
    this.mixer.update(delta);

    // Periodically rotate idle animation for variety
    if (this.currentName && this._isIdling()) {
      this._idleTimer += delta;
      if (this._idleTimer >= this._idleInterval) {
        this._idleTimer = 0;
        this._idleInterval = 6 + Math.random() * 6;
        this._playRandomIdle();
      }
    }
  }

  _isIdling() {
    // True if currently playing one of the idle pool clips
    return IDLE_POOL.includes(this.currentName);
  }

  _playRandomIdle() {
    // Pick a clip from the available idle pool, different from the current one
    const available = IDLE_POOL.filter(c => this.clips[c] && c !== this._lastIdleClip);
    if (!available.length) return;
    const pick = available[Math.floor(Math.random() * available.length)];
    this._lastIdleClip = pick;
    this._play(pick, true);
  }

  get availableGestures() {
    return Object.keys(GESTURE_CLIPS).filter(g =>
      GESTURE_CLIPS[g].some(c => this.clips[c])
    );
  }
}

export const GESTURE_NAMES = Object.keys(GESTURE_CLIPS);
