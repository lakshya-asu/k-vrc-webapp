const HINTS = [
  { label: 'Try saying', text: '"do a little dance"' },
  { label: 'Try saying', text: '"what are you thinking about?"' },
  { label: 'Fun fact',   text: '100 face states — every reply changes the expression' },
  { label: 'Try saying', text: '"tell me something surprising"' },
  { label: 'Fun fact',   text: 'Voice syncs live to audio amplitude' },
  { label: 'Controls',  text: 'Scroll to zoom · mouse to turn the head' },
  { label: 'Try saying', text: '"show me how you feel right now"' },
  { label: 'Try saying', text: '"explain the universe in one sentence"' },
];

export function initHints() {
  const panel   = document.getElementById('hints-panel');
  const labelEl = document.getElementById('hints-label');
  const textEl  = document.getElementById('hints-text');
  if (!panel) return;

  let idx = 0;
  let timer = null;
  let dismissed = false;

  function show(i) {
    const h = HINTS[i % HINTS.length];
    panel.classList.remove('visible');
    setTimeout(() => {
      if (dismissed) return;
      labelEl.textContent = h.label;
      textEl.textContent  = h.text;
      panel.classList.toggle('try-saying', h.label === 'Try saying');
      panel.classList.add('visible');
    }, 320);
  }

  function advance() {
    if (dismissed) return;
    idx = (idx + 1) % HINTS.length;
    show(idx);
    timer = setTimeout(advance, 5500);
  }

  function dismiss() {
    if (dismissed) return;
    dismissed = true;
    clearTimeout(timer);
    panel.classList.remove('visible');
  }

  // Click "Try saying" hint → paste into input
  panel.addEventListener('click', () => {
    const h = HINTS[idx % HINTS.length];
    if (h.label !== 'Try saying') return;
    const input = document.getElementById('chat-input');
    if (!input) return;
    input.value = h.text.replace(/^"|"$/g, '');
    input.focus();
  });

  // Initial reveal after loading clears
  timer = setTimeout(() => {
    if (dismissed) return;
    labelEl.textContent = HINTS[0].label;
    textEl.textContent  = HINTS[0].text;
    panel.classList.toggle('try-saying', HINTS[0].label === 'Try saying');
    panel.classList.add('visible');
    timer = setTimeout(advance, 5500);
  }, 2800);


  // Dismiss when user starts chatting
  document.getElementById('send-btn')?.addEventListener('click',     dismiss);
  document.getElementById('mic-btn')?.addEventListener('mousedown',  dismiss);
  document.getElementById('chat-input')?.addEventListener('keydown', e => {
    if (e.key === 'Enter') dismiss();
  });
  window.addEventListener('keydown', e => { if (e.code === 'KeyT') dismiss(); }, { once: true });
}
