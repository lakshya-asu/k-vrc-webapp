import { setEmotion, setSpeakingAmplitude } from './faceScreen.js';
import { applyEmotion } from './emotions.js';
import { fetchSidenote, hideSidenote } from './sidenote.js';

let history = [];
let ttsVoice = null;
let sceneRefs = null;
let robotRef = null;
let _ampInterval = null;

// ── TTS ──────────────────────────────────────────────────────
function setupTTS() {
  if (!window.speechSynthesis) return;
  const load = () => {
    const voices = window.speechSynthesis.getVoices();
    ttsVoice = voices.find(v => v.name.includes('Google') && v.lang.startsWith('en'))
            ?? voices.find(v => v.lang.startsWith('en'))
            ?? voices[0] ?? null;
  };
  window.speechSynthesis.addEventListener('voiceschanged', load);
  load();
}

function speak(text) {
  if (!window.speechSynthesis || !ttsVoice) return;
  window.speechSynthesis.cancel();
  const utt = new SpeechSynthesisUtterance(text);
  utt.voice = ttsVoice;
  utt.rate = 0.95;
  utt.pitch = 0.85;
  utt.onstart = () => {
    _ampInterval = setInterval(() => {
      setSpeakingAmplitude(0.3 + Math.random() * 0.7);
    }, 110);
  };
  utt.onboundary = () => setSpeakingAmplitude(0.5 + Math.random() * 0.5);
  utt.onend = () => { clearInterval(_ampInterval); setSpeakingAmplitude(0); };
  window.speechSynthesis.speak(utt);
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
  if (sceneRefs) {
    const cfg = applyEmotion(emotion, sceneRefs);
    robotRef?.setEmotionCfg(cfg);
    robotRef?.startBodyMotion(cfg.bodyMotion);
  }
  if (emotion === 'angry') robotRef?.triggerHeadJerk();
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

  let reply, emotion, sidenote_topic = null;
  try {
    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: trimmed, history: historySnapshot }),
    });
    if (res.status === 400) { addBubble('Message too long.', 'robot'); applyEmotionFull('neutral'); return; }
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    reply          = data.reply;
    emotion        = data.emotion ?? 'neutral';
    sidenote_topic = data.sidenote_topic ?? null;
    if (data.gesture) robotRef?.playGesture(data.gesture);
  } catch (err) {
    console.error(err);
    addBubble("K-VRC is offline. Try again?", 'robot');
    applyEmotionFull('sad');
    return;
  }

  history = [...history, { role: 'model', text: reply }].slice(-20);
  addBubble(reply, 'robot');
  applyEmotionFull(emotion);
  setTimeout(() => speak(reply), 300);

  if (sidenote_topic) fetchSidenote(sidenote_topic, trimmed);
  else hideSidenote();
}

// ── Mic ──────────────────────────────────────────────────────
function setupMic() {
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  const btn = document.getElementById('mic-btn');
  if (!SR) { btn?.style && (btn.style.display = 'none'); return; }
  const recog = new SR();
  recog.lang = 'en-US';
  recog.interimResults = false;
  let active = false;
  btn?.addEventListener('click', () => {
    active ? recog.stop() : recog.start();
    active = !active;
    btn.classList.toggle('active', active);
  });
  recog.addEventListener('result', e => {
    document.getElementById('chat-input').value = e.results[0][0].transcript;
    active = false;
    btn.classList.remove('active');
  });
  recog.addEventListener('end', () => { active = false; btn.classList.remove('active'); });
}

// ── Init ─────────────────────────────────────────────────────
export function initChat(robot, refs) {
  robotRef = robot;
  sceneRefs = refs;
  setupTTS();
  setupMic();

  const input = document.getElementById('chat-input');
  document.getElementById('send-btn')?.addEventListener('click', () => sendMessage(input.value));
  input?.addEventListener('keydown', e => { if (e.key === 'Enter' && !e.shiftKey) sendMessage(input.value); });

  addBubble("K-VRC online. What do you want.", 'robot');
  applyEmotionFull('neutral');
}
