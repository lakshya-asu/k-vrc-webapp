import { setEmotion, setSpeakingAmplitude } from './faceScreen.js';
import { applyEmotion } from './emotions.js';
import { fetchSidenote, hideSidenote } from './sidenote.js';

let history = [];
let sceneRefs = null;
let robotRef = null;

// ── TTS ──────────────────────────────────────────────────────
let _audioCtx = null;
let _currentSource = null;

function getAudioCtx() {
  if (!_audioCtx || _audioCtx.state === 'closed') _audioCtx = new AudioContext();
  return _audioCtx;
}

async function speak(text, emotion = 'neutral') {
  // Stop any playing audio
  if (_currentSource) { try { _currentSource.stop(); } catch (_) {} _currentSource = null; }
  setSpeakingAmplitude(0);

  let arrayBuffer;
  try {
    const res = await fetch('/api/tts', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text, emotion }),
    });
    if (!res.ok) return;
    arrayBuffer = await res.arrayBuffer();
  } catch { return; }

  const ctx = getAudioCtx();
  if (ctx.state === 'suspended') await ctx.resume();

  let audioBuffer;
  try { audioBuffer = await ctx.decodeAudioData(arrayBuffer); } catch { return; }

  const analyser = ctx.createAnalyser();
  analyser.fftSize = 256;
  const dataArray = new Uint8Array(analyser.frequencyBinCount);

  const source = ctx.createBufferSource();
  source.buffer = audioBuffer;
  source.connect(analyser);
  analyser.connect(ctx.destination);
  _currentSource = source;

  let rafId;
  const tick = () => {
    analyser.getByteFrequencyData(dataArray);
    const avg = dataArray.reduce((a, b) => a + b, 0) / dataArray.length;
    setSpeakingAmplitude(avg / 128);
    rafId = requestAnimationFrame(tick);
  };

  source.onended = () => {
    cancelAnimationFrame(rafId);
    setSpeakingAmplitude(0);
    _currentSource = null;
  };

  source.start();
  tick();
}

// ── Chat UI ──────────────────────────────────────────────────
function addBubble(text, role) {
  const log = document.getElementById('chat-log');
  const div = document.createElement('div');
  div.className = `bubble ${role === 'user' ? 'user' : 'robot'}`;
  div.textContent = text;
  log.appendChild(div);
  log.scrollTop = log.scrollHeight;
}

function showTyping() {
  const log = document.getElementById('chat-log');
  const div = document.createElement('div');
  div.className = 'bubble robot typing-bubble';
  div.id = 'typing-indicator';
  div.innerHTML = '<span class="typing-dot"></span><span class="typing-dot"></span><span class="typing-dot"></span>';
  log.appendChild(div);
  log.scrollTop = log.scrollHeight;
}

function hideTyping() {
  document.getElementById('typing-indicator')?.remove();
}

const EMOTION_UI = {
  neutral:  { dot: '#00e5ff', bg: 'radial-gradient(ellipse 80% 60% at 50% 40%, rgba(0,229,255,0.05) 0%, transparent 70%)' },
  happy:    { dot: '#00ff88', bg: 'radial-gradient(ellipse 80% 60% at 50% 40%, rgba(0,255,136,0.07) 0%, transparent 70%)' },
  excited:  { dot: '#ffe600', bg: 'radial-gradient(ellipse 80% 60% at 50% 40%, rgba(255,230,0,0.06) 0%, transparent 70%)' },
  sad:      { dot: '#4488ff', bg: 'radial-gradient(ellipse 80% 60% at 50% 40%, rgba(68,136,255,0.07) 0%, transparent 70%)' },
  angry:    { dot: '#ff2200', bg: 'radial-gradient(ellipse 80% 60% at 50% 40%, rgba(255,34,0,0.08) 0%, transparent 70%)' },
  thinking: { dot: '#bb44ff', bg: 'radial-gradient(ellipse 80% 60% at 50% 40%, rgba(187,68,255,0.06) 0%, transparent 70%)' },
};

function updateEmotionUI(emotion) {
  const ui = EMOTION_UI[emotion] ?? EMOTION_UI.neutral;
  const dot = document.getElementById('emotion-dot');
  const label = document.getElementById('emotion-label');
  const bg = document.getElementById('bg-layer');
  if (dot) { dot.style.background = ui.dot; dot.style.boxShadow = `0 0 10px ${ui.dot}`; }
  if (label) label.textContent = emotion;
  if (bg) bg.style.background = ui.bg;
}

function applyEmotionFull(emotion) {
  setEmotion(emotion);
  updateEmotionUI(emotion);
  robotRef?.setEmotionName(emotion);
  if (sceneRefs) {
    const cfg = applyEmotion(emotion, sceneRefs);
    robotRef?.setEmotionCfg(cfg);
    robotRef?.startBodyMotion(cfg.bodyMotion);
  }
  if (emotion === 'angry') robotRef?.triggerHeadJerk();
  if (emotion === 'thinking') {
    robotRef?.setExpression('thinking_default');
    const label = document.getElementById('emotion-label');
    if (label) label.textContent = 'thinking_default';
  }
}

// ── Send message ─────────────────────────────────────────────
async function sendMessage(text) {
  const trimmed = text.trim();
  if (!trimmed) return;

  const input = document.getElementById('chat-input');
  const sendBtn = document.getElementById('send-btn');
  if (sendBtn.disabled) return;
  input.value = '';
  sendBtn.disabled = true;
  setTimeout(() => { sendBtn.disabled = false; }, 1000);

  addBubble(trimmed, 'user');
  const historySnapshot = history.slice(-20);
  history = [...history, { role: 'user', text: trimmed }].slice(-20);
  applyEmotionFull('thinking');
  showTyping();

  let reply, emotion, data = null, sidenote_topic = null;
  try {
    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: trimmed, history: historySnapshot }),
    });
    hideTyping();
    if (res.status === 400) { addBubble('Message too long.', 'robot'); applyEmotionFull('neutral'); return; }
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    data = await res.json();
    reply          = data.reply;
    emotion        = data.emotion ?? 'neutral';
    sidenote_topic = data.sidenote_topic ?? null;
    if (data.gesture) robotRef?.playGesture(data.gesture);
    if (data.infer_result) robotRef?.applyInferResult(data.infer_result);
  } catch (err) {
    hideTyping();
    console.error(err);
    addBubble("K-VRC is offline. Try again?", 'robot');
    applyEmotionFull('sad');
    return;
  }

  history = [...history, { role: 'model', text: reply }].slice(-20);
  addBubble(reply, 'robot');
  applyEmotionFull(emotion);
  const expr = data.expression ?? 'neutral_idle';
  robotRef?.setExpression(expr);
  const label = document.getElementById('emotion-label');
  if (label) label.textContent = expr;
  setTimeout(() => speak(reply, emotion), 300);

  if (sidenote_topic) fetchSidenote(sidenote_topic, trimmed);
  else hideSidenote();
}

// ── Mic (MediaRecorder + Whisper STT) ────────────────────────
let _micStream = null;
let _mediaRecorder = null;
let _audioChunks = [];
let _micAnalyser = null;
let _micAnimId = null;
let _micActive = false;
let _micBusy = false;

function _animateMicBars() {
  if (!_micActive || !_micAnalyser) return;
  const data = new Uint8Array(_micAnalyser.frequencyBinCount);
  const bars = document.querySelectorAll('.mic-bar');
  // pick 4 representative frequency bins spread across the spectrum
  const picks = [0.08, 0.22, 0.42, 0.65];
  function frame() {
    if (!_micActive) return;
    _micAnimId = requestAnimationFrame(frame);
    _micAnalyser.getByteFrequencyData(data);
    bars.forEach((bar, i) => {
      const val = data[Math.floor(picks[i] * data.length)] / 255;
      bar.style.height = `${3 + val * 17}px`;
    });
  }
  frame();
}

function _stopMicBars() {
  if (_micAnimId) { cancelAnimationFrame(_micAnimId); _micAnimId = null; }
  document.querySelectorAll('.mic-bar').forEach(b => { b.style.height = '3px'; });
}

async function _blobToBase64(blob) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result.split(',')[1]);
    reader.onerror = reject;
    reader.readAsDataURL(blob);
  });
}

function setupMic() {
  const btn = document.getElementById('mic-btn');
  if (!btn) return;
  if (!navigator.mediaDevices?.getUserMedia || !window.MediaRecorder) {
    btn.style.display = 'none';
    return;
  }

  async function startRecording() {
    if (_micActive || _micBusy) return;
    let stream;
    try { stream = await navigator.mediaDevices.getUserMedia({ audio: true }); }
    catch { console.warn('Mic access denied'); return; }

    _micStream = stream;
    _micActive = true;
    btn.classList.add('active');

    const actx = new AudioContext();
    const src = actx.createMediaStreamSource(stream);
    _micAnalyser = actx.createAnalyser();
    _micAnalyser.fftSize = 32;
    src.connect(_micAnalyser);

    _audioChunks = [];
    _mediaRecorder = new MediaRecorder(stream);
    _mediaRecorder.ondataavailable = e => { if (e.data.size > 0) _audioChunks.push(e.data); };
    _mediaRecorder.start(100);
    _animateMicBars();
  }

  async function stopRecording() {
    if (!_micActive) return;
    _micActive = false;
    _micBusy = true;
    btn.classList.remove('active');
    btn.classList.add('transcribing');
    _stopMicBars();

    const stopped = new Promise(resolve => { _mediaRecorder.onstop = resolve; });
    _mediaRecorder.stop();
    _micStream?.getTracks().forEach(t => t.stop());
    await stopped;

    const blob = new Blob(_audioChunks, { type: _mediaRecorder.mimeType || 'audio/webm' });
    try {
      const base64 = await _blobToBase64(blob);
      const res = await fetch('/api/stt', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ audio: base64, mimeType: blob.type }),
      });
      if (res.ok) {
        const { text } = await res.json();
        if (text?.trim()) sendMessage(text.trim());
      }
    } catch (err) {
      console.error('STT error:', err);
    } finally {
      _micBusy = false;
      btn.classList.remove('transcribing');
    }
  }

  // Hold button to record
  btn.addEventListener('mousedown', e => { e.preventDefault(); startRecording(); });
  btn.addEventListener('mouseup', () => stopRecording());
  btn.addEventListener('mouseleave', () => { if (_micActive) stopRecording(); });

  // Touch (mobile)
  btn.addEventListener('touchstart', e => { e.preventDefault(); startRecording(); }, { passive: false });
  btn.addEventListener('touchend', () => stopRecording());

  // Hold T to record — skipped when chat input is focused
  window.addEventListener('keydown', e => {
    if (e.code !== 'KeyT' || e.repeat) return;
    if (document.activeElement?.id === 'chat-input') return;
    e.preventDefault();
    startRecording();
  });
  window.addEventListener('keyup', e => {
    if (e.code !== 'KeyT') return;
    stopRecording();
  });
}

// ── Init ─────────────────────────────────────────────────────
export function initChat(robot, refs) {
  robotRef = robot;
  sceneRefs = refs;
  setupMic();

  const input = document.getElementById('chat-input');
  document.getElementById('send-btn')?.addEventListener('click', () => sendMessage(input.value));
  input?.addEventListener('keydown', e => { if (e.key === 'Enter' && !e.shiftKey) sendMessage(input.value); });

  // Collapse toggle
  const overlay = document.getElementById('chat-overlay');
  document.getElementById('chat-header')?.addEventListener('click', () => {
    overlay?.classList.toggle('collapsed');
  });

  addBubble("K-VRC online. What do you want.", 'robot');
  applyEmotionFull('neutral');
}
