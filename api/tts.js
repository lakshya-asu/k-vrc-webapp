// api/tts.js — OpenAI TTS proxy
const MAX_TEXT_LEN = 1000;

export default async function handler(req, res) {
  // CORS — identical to api/chat.js
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

  const { text } = req.body || {};
  if (!text || typeof text !== 'string' || text.trim().length === 0) {
    return res.status(400).json({ error: 'text is required' });
  }
  if (text.length > MAX_TEXT_LEN) {
    return res.status(400).json({ error: 'Text too long' });
  }

  const openaiKey = process.env.OPENAI_API_KEY;
  if (!openaiKey) {
    console.error('OPENAI_API_KEY not set');
    return res.status(500).json({ error: 'TTS not configured' });
  }

  try {
    const ttsRes = await fetch('https://api.openai.com/v1/audio/speech', {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${openaiKey}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        model: 'tts-1',
        input: text.trim(),
        voice: 'onyx',
        response_format: 'mp3',
      }),
    });

    if (!ttsRes.ok) {
      const err = await ttsRes.text();
      console.error('OpenAI TTS error:', ttsRes.status, err);
      return res.status(500).json({ error: 'TTS service error' });
    }

    const audioData = await ttsRes.arrayBuffer();
    res.setHeader('Content-Type', 'audio/mpeg');
    res.setHeader('Content-Length', audioData.byteLength);
    return res.end(Buffer.from(audioData));
  } catch (err) {
    console.error('TTS fetch error:', err.message);
    return res.status(500).json({ error: 'TTS request failed' });
  }
}
