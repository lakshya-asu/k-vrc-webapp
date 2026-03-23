// ── Sidenote panel ───────────────────────────────────────────
// Owns DOM for #sidenote-panel. Called from chat.js fire-and-forget.

function showSidenote(text, links) {
  const panel = document.getElementById('sidenote-panel');
  const textEl = document.getElementById('sidenote-text');
  const linksEl = document.getElementById('sidenote-links');
  if (!panel || !textEl || !linksEl) return;

  textEl.textContent = text;
  linksEl.innerHTML = '';
  links.forEach(({ label, url }) => {
    const a = document.createElement('a');
    a.href = url;
    a.textContent = label;
    a.target = '_blank';
    a.rel = 'noopener noreferrer';
    linksEl.appendChild(a);
  });

  panel.classList.add('visible');
}

export function hideSidenote() {
  const panel = document.getElementById('sidenote-panel');
  if (panel) panel.classList.remove('visible');
}

export async function fetchSidenote(topic, message) {
  try {
    const res = await fetch('/api/sidenote', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ topic, message }),
    });

    if (!res.ok) { hideSidenote(); return; }

    const data = await res.json();
    if (!data.text) { hideSidenote(); return; }

    showSidenote(data.text, data.links ?? []);
  } catch {
    hideSidenote();
  }
}
