// src/agent/agentLoop.js
// Browser glue for agent mode: feeds real user input into the
// deterministic behavior brain and applies its actions to the robot
// and the face screen. The brain itself (behaviorBrain.js) is pure
// and testable; everything DOM- or three-flavored lives here.
//
// Enabled by default; disable with ?agent=off. Reproduce a session
// with ?agentseed=<int>. Debug handle: window.KVRC_AGENT.
//
// A model-driven brain (WebLLM) is a planned stretch goal: anything
// that implements { tick(nowMs), notify(event, nowMs) } can be
// swapped in behind a ?brain= flag. Only the deterministic brain
// ships today. See docs/agent-mode.md.

import * as THREE from 'three';
import { createBrain } from './behaviorBrain.js';
import { setFaceGlyph, clearFaceGlyph } from '../faceScreen.js';

const CURSOR_NOTIFY_MS = 200;   // proximity update cadence
const TYPING_NOTIFY_MS = 1200;  // typing event cadence while keys land

export function initAgent({ robot, camera }) {
  const params = new URLSearchParams(window.location.search);
  const seedParam = Number.parseInt(params.get('agentseed') ?? '', 10);
  const seed = Number.isFinite(seedParam) ? seedParam : (Date.now() >>> 0);
  const brain = createBrain({ seed });

  // The brain owns idle cadence now; stop the built-in randomizers.
  robot.setIdleAutonomy({ body: false, face: false });

  const state = {
    seed,
    enabled: true,
    lastCursorNotify: 0,
    lastTypingNotify: 0,
    lastProximity: 0,
    actionLog: [],       // last 50 applied actions, for debugging/verification
  };

  const _headPos = new THREE.Vector3();

  function headScreenPosition() {
    const head = robot.getHeadBone();
    if (!head) return null;
    // Refresh matrices ourselves: mousemove can fire before the first
    // rendered frame (or while the tab is hidden), when matrixWorld
    // and the camera inverse are stale.
    head.updateWorldMatrix(true, false);
    camera.updateMatrixWorld();
    head.getWorldPosition(_headPos);
    _headPos.project(camera);
    if (!Number.isFinite(_headPos.x) || !Number.isFinite(_headPos.y)) return null;
    return {
      x: (_headPos.x * 0.5 + 0.5) * window.innerWidth,
      y: (-_headPos.y * 0.5 + 0.5) * window.innerHeight,
    };
  }

  function cursorProximity(clientX, clientY) {
    const head = headScreenPosition();
    if (!head) return 0;
    const dx = clientX - head.x;
    const dy = clientY - head.y;
    const dist = Math.sqrt(dx * dx + dy * dy);
    const radius = 0.30 * Math.min(window.innerWidth, window.innerHeight);
    return Math.max(0, 1 - dist / (radius * 2));
  }

  function apply(actions) {
    for (const action of actions) {
      switch (action.type) {
        case 'gesture':
          robot.playGesture(action.name);
          break;
        case 'expression': {
          robot.setExpression(action.name);
          const label = document.getElementById('emotion-label');
          if (label) label.textContent = action.name;
          break;
        }
        case 'glyph': {
          setFaceGlyph(action.spec);
          const label = document.getElementById('emotion-label');
          if (label) label.textContent = `glyph:${action.spec.text ?? action.spec.eyes}`;
          break;
        }
        case 'clearGlyph':
          clearFaceGlyph();
          break;
        case 'gaze':
          robot.setGaze(action.yawDeg, action.pitchDeg);
          break;
        case 'headJerk':
          robot.triggerHeadJerk();
          break;
        default:
          break;
      }
      state.actionLog.push({ t: Math.round(performance.now()), ...action });
      if (state.actionLog.length > 50) state.actionLog.shift();
    }
  }

  function notify(event) {
    if (!state.enabled) return;
    apply(brain.notify(event, performance.now()));
  }

  // ── Cursor proximity ───────────────────────────────────────
  window.addEventListener('mousemove', (e) => {
    const now = performance.now();
    if (now - state.lastCursorNotify < CURSOR_NOTIFY_MS) return;
    state.lastCursorNotify = now;
    const proximity = cursorProximity(e.clientX, e.clientY);
    state.lastProximity = proximity;
    notify({ type: 'cursor', proximity });
  });

  // ── Pokes: clicks on the 3D canvas near the robot ──────────
  document.getElementById('three-canvas')?.addEventListener('click', (e) => {
    const proximity = cursorProximity(e.clientX, e.clientY);
    if (proximity >= 0.55) notify({ type: 'poke' });
  });

  // ── Typing in the chat box ─────────────────────────────────
  document.getElementById('chat-input')?.addEventListener('input', () => {
    const now = performance.now();
    if (now - state.lastTypingNotify < TYPING_NOTIFY_MS) return;
    state.lastTypingNotify = now;
    notify({ type: 'typing' });
  });

  // ── Chat lifecycle events (dispatched by chat.js) ──────────
  window.addEventListener('kvrc:user-message', () => notify({ type: 'user_message' }));
  window.addEventListener('kvrc:reply', (e) => notify({ type: 'reply', emotion: e.detail?.emotion }));
  window.addEventListener('kvrc:reply-error', () => notify({ type: 'reply_error' }));

  // ── Tab visibility ─────────────────────────────────────────
  document.addEventListener('visibilitychange', () => {
    notify({ type: document.hidden ? 'hidden' : 'visible' });
  });

  const agent = {
    brain,
    state,
    robot,
    /** Call once per rendered frame. */
    update() {
      if (!state.enabled) return;
      apply(brain.tick(performance.now()));
    },
    setEnabled(enabled) {
      state.enabled = !!enabled;
      robot.setIdleAutonomy({ body: !enabled, face: !enabled });
      if (!enabled) clearFaceGlyph();
    },
  };

  // Debug and verification handle.
  window.KVRC_AGENT = agent;
  console.log(`K-VRC agent mode on (seed ${seed}). Disable with ?agent=off`);
  return agent;
}
