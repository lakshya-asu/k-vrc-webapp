// api/stt.js — OpenAI Whisper STT proxy
export default async function handler(req, res) {
  // CORS
  const ALLOWED_ORIGIN = process.env.ALLOWED_ORIGIN;
  const origin = req.headers.origin || '';
  const isLocalDev = origin.startsWith('http://localhost') || origin.startsWith('http://127.0.0.1');
  const isVercel = origin.endsWith('.vercel.app');
  const isCustom = ALLOWED_ORIGIN && origin === ALLOWED_ORIGIN;
  if (!isLocalDev && !isVercel && !isCustom) return res.status(403).json({ error: 'Forbidden' });

  res.setHeader('Access-Control-Allow-Origin', origin);
  res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

  if (req.method === 'OPTIONS') return res.status(200).end();
  if (req.method !== 'POST') return res.status(405).json({ error: 'Method not allowed' });

  const { audio, mimeType } = req.body || {};
  if (!audio || typeof audio !== 'string') {
    return res.status(400).json({ error: 'audio is required' });
  }

  const openaiKey = process.env.OPENAI_API_KEY;
  if (!openaiKey) {
    console.error('OPENAI_API_KEY not set');
    return res.status(500).json({ error: 'STT not configured' });
  }

  try {
    const audioBuffer = Buffer.from(audio, 'base64');
    const type = mimeType || 'audio/webm';
    let ext = 'webm';
    if (type.includes('ogg'))             ext = 'ogg';
    else if (type.includes('mp4') || type.includes('m4a')) ext = 'mp4';
    else if (type.includes('wav'))        ext = 'wav';

    const form = new FormData();
    form.append('file', new Blob([audioBuffer], { type }), `audio.${ext}`);
    form.append('model', 'whisper-1');
    form.append('language', 'en');

    const whisperRes = await fetch('https://api.openai.com/v1/audio/transcriptions', {
      method: 'POST',
      headers: { 'Authorization': `Bearer ${openaiKey}` },
      body: form,
    });

    if (!whisperRes.ok) {
      const err = await whisperRes.text();
      console.error('Whisper STT error:', whisperRes.status, err);
      return res.status(500).json({ error: 'STT service error' });
    }

    const data = await whisperRes.json();
    return res.status(200).json({ text: data.text ?? '' });
  } catch (err) {
    console.error('STT fetch error:', err.message);
    return res.status(500).json({ error: 'STT request failed' });
  }
}
