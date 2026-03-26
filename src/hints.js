const HINTS = [
  { label: 'Try saying',  text: '"do a little dance"' },
  { label: 'Try saying',  text: '"what are you thinking about?"' },
  { label: 'Fun fact',    text: '100 face states — every reply changes the expression' },
  { label: 'Try saying',  text: '"tell me something surprising"' },
  { label: 'Fun fact',    text: 'Voice trained on real audio — mouth syncs live to amplitude' },
  { label: 'Controls',   text: 'Scroll to zoom · move the mouse to turn the head' },
  { label: 'Try saying',  text: '"show me how you feel right now"' },
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

  // Initial reveal after loading clears
  timer = setTimeout(() => {
    if (dismissed) return;
    labelEl.textContent = HINTS[0].label;
    textEl.textContent  = HINTS[0].text;
    panel.classList.add('visible');
    timer = setTimeout(advance, 5500);
  }, 2800);


  // Dismiss when user starts chatting
  document.getElementById('send-btn')?.addEventListener('click',     dismiss);
  document.getElementById('mic-btn')?.addEventListener('mousedown',  dismiss);
  document.getElementById('chat-input')?.addEventListener('keydown', e => {
    if (e.key === 'Enter') dismiss();
  });
  window.addEventListener('keydown', e => { if (e.code === 'Backquote') dismiss(); }, { once: true });
}
